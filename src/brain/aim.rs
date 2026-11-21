use super::math::*;
use super::memory::Track;
use super::{Brain, Mode, Target};
use crate::io::proto::{ActorKind, Obs, SoundDist, WeaponKind, WeaponObs};
use crate::sim::world::World;

#[derive(Clone, Debug)]
pub(crate) struct AimTarget {
    pub feet: V3,
    pub vel: V3,
    pub accel: V3,
    pub radius: f64,
    pub height: f64,
    pub hp: f64,
    pub fixed: bool,
}

pub(crate) struct FirePlan {
    pub aim: V2,
    pub fire: [bool; 2],
    pub reload: bool,
    pub target: Option<AimTarget>,
    pub armed: bool,
    pub charging: bool,
}

impl Brain {
    pub(crate) fn aim_target(&self, track: &Track) -> AimTarget {
        let (radius, height) = if track.kind == ActorKind::Npc {
            (self.config.npc.radius, self.config.npc.height)
        } else {
            let s = 1.0 + self.config.cubes.size_per_cube * track.cubes.max(0.0);
            (self.config.mech.radius * s, self.config.mech.height * s)
        };
        AimTarget { feet: track.pos, vel: track.vel, accel: track.accel, radius, height, hp: track.hp, fixed: false }
    }

    fn idle_aim(&self, obs: &Obs) -> V2 {
        let t = obs.t;
        if let Some(u) = &self.memory.unseen {
            if t - u.time < 3.0 {
                return [u.bearing, 0.0];
            }
        }
        let mut best: Option<(f64, f64)> = None;
        for s in &self.memory.sounds {
            if t - s.time > 2.0 {
                continue;
            }
            let k = match s.kind.as_str() {
                "gun" | "laser" | "laser_charge" | "lightning" | "mortar" | "step" | "boost" | "jet" | "reload" => 1.0,
                _ => 0.0,
            };
            let d = match s.dist {
                SoundDist::Near => 3.0,
                SoundDist::Mid => 1.5,
                _ => 0.0,
            };
            let pr = k * d - (t - s.time) * 0.3;
            if k * d > 0.0 && best.map_or(true, |b| pr > b.0) {
                best = Some((pr, s.bearing));
            }
        }
        if let Some((_, b)) = best {
            return [b, 0.0];
        }
        let base = if self.last_dir[0].hypot(self.last_dir[1]) > 0.1 { self.last_dir[1].atan2(self.last_dir[0]) } else { obs.me.aim[0] };
        let ph = 2.0 * std::f64::consts::PI * t / self.params.scan_period.max(0.5);
        [normalize_angle(base + self.params.scan_amp * ph.sin()), 0.0]
    }

    fn pick_fire_target(&self, obs: &Obs, world: &World, muzzle: V3) -> Option<(AimTarget, bool)> {
        let t = obs.t;
        let me = &obs.me;
        let mut max_range: f64 = 0.0;
        for w in me.weapons.iter().flatten() {
            let r = match w.kind {
                WeaponKind::Gun => w.stats.range * self.params.gun_range_frac,
                WeaponKind::Mortar => {
                    if w.stats.gravity > 0.0 {
                        w.stats.speed * w.stats.speed / w.stats.gravity * 0.95
                    } else {
                        0.0
                    }
                }
                _ => w.stats.range,
            };
            max_range = max_range.max(r);
        }
        let mut best: Option<(f64, AimTarget, bool)> = None;
        let held = self.is_holding(t);
        for tr in &self.memory.tracks {
            if held || tr.kind != ActorKind::Mech || t - tr.last_seen > 0.15 {
                continue;
            }
            if self.roof_hold && t - tr.last_damage_time > 1.5 {
                continue;
            }
            let at = self.aim_target(tr);
            let c = raised(at.feet, at.height * 0.5);
            let d = dist(c, muzzle);
            let clear = world.ray(muzzle, c).is_none();
            let attacker = t - tr.last_damage_time < 3.0;
            let mut s = -0.3 * d + 40.0 * (1.0 - clamped_ratio(tr.hp, tr.max_hp));
            if clear {
                s += 100.0;
            }
            if attacker {
                s += 40.0;
            }
            if self.target == Target::Mech(tr.id) {
                s += 50.0;
            }
            if d < max_range {
                s += 50.0;
            }
            let to_us = (me.pos[1] - tr.pos[1]).atan2(me.pos[0] - tr.pos[0]);
            let aligned = angle_diff(tr.aim[0], to_us).abs() < 0.5;
            let allow = self.mode != Mode::Retreat || attacker || (aligned && clear) || d < 60.0;
            if best.as_ref().map_or(true, |b| s > b.0) {
                best = Some((s, at, allow));
            }
        }
        if let Some((_, at, allow)) = best {
            return Some((at, allow));
        }
        let mut bn: Option<(f64, AimTarget, bool)> = None;
        for tr in &self.memory.tracks {
            if tr.kind != ActorKind::Npc || t - tr.last_seen > 0.15 {
                continue;
            }
            let attacker = t - tr.last_damage_time < 3.0;
            let hunted = self.target == Target::Npc(tr.id);
            if (!attacker && !hunted) || (self.roof_hold && t - tr.last_damage_time > 1.5) {
                continue;
            }
            let d = dist(tr.pos, me.pos);
            let s = -d + if hunted { 50.0 } else { 0.0 };
            if bn.as_ref().map_or(true, |b| s > b.0) {
                bn = Some((s, self.aim_target(tr), self.mode != Mode::Retreat || attacker));
            }
        }
        if let Some((_, at, allow)) = bn {
            return Some((at, allow));
        }
        if !self.allow_crates {
            return None;
        }
        let mut bc: Option<(f64, AimTarget)> = None;
        let mut rays = 0;
        for it in &self.memory.crates {
            if t - it.last_seen > 0.3 || it.hp <= 0.0 || self.memory.blacklisted(it.id, t) {
                continue;
            }
            let c = raised(it.pos, self.config.crates.height * 0.5);
            let d = dist(c, muzzle);
            if d > max_range.min(170.0) {
                continue;
            }
            let s = -d + if self.target == Target::Crate(it.id) { 60.0 } else { 0.0 };
            if bc.as_ref().map_or(true, |b| s > b.0) && rays < 8 {
                rays += 1;
                if world.ray(muzzle, c).is_none() {
                    let at = AimTarget { feet: it.pos, vel: [0.0; 3], accel: [0.0; 3], radius: self.config.crates.radius, height: self.config.crates.height, hp: it.hp, fixed: true };
                    bc = Some((s, at));
                }
            }
        }
        bc.map(|b| (b.1, true))
    }

    fn mortar_solve(&self, obs: &Obs, world: &World, tg: &AimTarget, w: &WeaponObs) -> Option<V2> {
        let p = &self.params;
        let mech = &obs.me.stats;
        let tube = raised(obs.me.pos, mech.height + 0.5);
        let v = w.stats.speed;
        let g = w.stats.gravity;
        if !(v > 0.0) || !(g > 0.0) {
            return None;
        }
        let horiz_speed = [tg.vel[0], tg.vel[1], 0.0];
        let speed = horiz_speed[0].hypot(horiz_speed[1]);
        if speed > p.mortar_max_target_speed {
            return None;
        }
        for high in [false, true] {
            let mut tf = horiz_dist(tg.feet, tube) / (v * 0.9);
            let mut sol: Option<(f64, V3, f64)> = None;
            for _ in 0..3 {
                let track_pos = add(raised(tg.feet, 0.5), scale(horiz_speed, tf + p.latency));
                let d = horiz_dist(track_pos, tube);
                match mortar_solution(d, track_pos[2] - tube[2], v, g, high) {
                    Some((pitch, t2)) => {
                        tf = t2;
                        sol = Some((pitch, track_pos, d));
                    }
                    None => {
                        sol = None;
                        break;
                    }
                }
            }
            let Some((pitch, tp, d)) = sol else { continue };
            if d < p.mortar_min_range || pitch < mech.min_pitch + 0.01 || pitch > mech.max_pitch - 0.01 {
                continue;
            }
            if tf > p.mortar_max_flight && speed > 2.0 && !tg.fixed {
                continue;
            }
            if w.stats.ttl > 0.0 && tf > w.stats.ttl * 0.95 {
                continue;
            }
            let yaw = (tp[1] - tube[1]).atan2(tp[0] - tube[0]);
            let vel = scale(aim_dir(yaw, pitch), v);
            let mut prev = tube;
            let mut clear = true;
            let n = 10;
            for k in 1..=n {
                let tt = tf * 0.93 * k as f64 / n as f64;
                let q = [tube[0] + vel[0] * tt, tube[1] + vel[1] * tt, tube[2] + vel[2] * tt - 0.5 * g * tt * tt];
                if world.ray(prev, q).is_some() {
                    clear = false;
                    break;
                }
                prev = q;
            }
            if clear {
                return Some([yaw, pitch]);
            }
        }
        None
    }

    pub(crate) fn fire_control(&mut self, obs: &Obs, world: &World) -> FirePlan {
        let p = self.params.clone();
        let me = &obs.me;
        let t = obs.t;
        let st = &me.stats;
        let muzzle = raised(me.pos, st.muzzle_height);
        let mut out = FirePlan { aim: self.idle_aim(obs), fire: [false; 2], reload: false, target: None, armed: false, charging: false };
        let mut tgt = self.pick_fire_target(obs, world, muzzle);
        if let Some(u) = &self.memory.unseen {
            let attacked_by_tgt = tgt.is_some() && self.memory.tracks.iter().any(|tr| t - tr.last_damage_time < 1.5 && t - tr.last_seen < 0.3);
            if t - u.time < 1.2 && !attacked_by_tgt {
                if let Some((tg, _)) = &tgt {
                    let c = raised(tg.feet, tg.height * 0.5);
                    if world.ray(muzzle, c).is_some() || dist(c, muzzle) > 60.0 {
                        tgt = None;
                    }
                }
                if tgt.is_none() {
                    out.aim = [u.bearing, 0.0];
                }
            }
        }

        if self.is_holding(t) {
            let mut face: Option<(f64, V3)> = None;
            for tr in &self.memory.tracks {
                if tr.kind != ActorKind::Mech || t - tr.last_seen > 2.5 {
                    continue;
                }
                let track_pos = tr.pos_at(t);
                let d = dist(track_pos, me.pos);
                let s = d - if t - tr.last_damage_time < 3.0 { 200.0 } else { 0.0 };
                if t - tr.last_damage_time < 3.0 && face.map_or(true, |f| s < f.0) {
                    face = Some((s, track_pos));
                }
            }
            if let Some((_, tp)) = face {
                tgt = None;
                out.aim = dir_to_aim(sub(raised(tp, self.config.mech.height * 0.5), muzzle));
            }
        }
        let Some((tg, allow)) = tgt else {
            for w in me.weapons.iter().flatten() {
                if w.kind == WeaponKind::Gun && w.reload <= 0.0 && w.ammo < w.stats.mag {
                    if w.ammo <= w.stats.mag * p.reload_frac || t - self.last_target_time > 1.5 {
                        out.reload = true;
                    }
                }
            }
            return out;
        };
        self.last_target_time = t;
        let c = raised(tg.feet, tg.height * 0.5);
        let rel = sub(c, muzzle);
        let d = len(rel).max(0.1);
        let direct = dir_to_aim(add(rel, scale(tg.vel, p.laser_latency)));
        let line_of_sight = world.ray(muzzle, c).is_none();
        let tol_y = (tg.radius * p.aim_tol).atan2(d);
        let tol_p = (tg.height * 0.5 * p.aim_tol).atan2(d);

        let mut gun: Option<(usize, V2, bool)> = None;
        let mut laser: Option<usize> = None;
        let mut lightning: Option<usize> = None;
        let mut mortar: Option<(usize, V2)> = None;
        for (i, w) in me.weapons.iter().enumerate() {
            let Some(w) = w else { continue };
            match w.kind {
                WeaponKind::Gun => {
                    let acc = if tg.fixed { [0.0; 3] } else { tg.accel };
                    if let Some((pt, _)) = lead_point(rel, tg.vel, acc, w.stats.speed.max(1.0), p.latency, p.acc_gain, p.acc_time) {
                        let ok = len(pt) <= w.stats.range * p.gun_range_frac && world.ray(muzzle, add(muzzle, pt)).is_none();
                        gun = Some((i, dir_to_aim(pt), ok));
                    }
                }
                WeaponKind::Laser => laser = Some(i),
                WeaponKind::Lightning => lightning = Some(i),
                WeaponKind::Mortar => {
                    if w.cooldown <= 0.25 && allow {
                        mortar = self.mortar_solve(obs, world, &tg, w).map(|a| (i, a));
                    }
                    if w.cooldown <= 1.0 {
                        out.armed = true;
                    }
                }
                WeaponKind::Unknown => {}
            }
        }

        let laser_w = laser.and_then(|i| me.weapons[i].as_ref().map(|w| (i, w)));
        let laser_in = laser_w.map_or(false, |(_, w)| d <= w.stats.range * 0.98 && line_of_sight);
        let laser_charging = laser_w.map_or(false, |(_, w)| w.charge > 0.0);
        let gun_ok = gun.map_or(false, |g| g.2);
        let mut command = match gun {
            Some((_, a, _)) => a,
            None => direct,
        };
        let mut laser_final = false;
        if let Some((_, w)) = laser_w {
            let full = w.stats.charge.max(0.05);
            if laser_charging && w.charge >= full - 0.2 {
                command = direct;
                laser_final = true;
            } else if !gun_ok && (laser_charging || laser_in) {
                command = direct;
            }
        }
        let mut mortar_primary = false;
        if !laser_final {
            if let Some((i, a)) = mortar {
                if me.weapons[i].as_ref().map_or(false, |w| w.cooldown <= 0.11) || !gun_ok {
                    command = a;
                    mortar_primary = true;
                }
            }
        }
        if !gun_ok && !mortar_primary && !laser_final {
            if let Some(i) = lightning {
                if me.weapons[i].as_ref().map_or(false, |w| d <= w.stats.range * 1.3) {
                    command = direct;
                }
            }
        }
        let pred = slew_angles(me.aim, command, st.turn_rate * 0.05, st.pitch_rate * 0.05);
        let on = |a: V2, ty: f64, tp: f64| angle_diff(pred[0], a[0]).abs() * a[1].cos().max(0.2) <= ty && (pred[1] - a[1]).abs() <= tp;

        if let Some((i, a, ok)) = gun {
            if let Some(w) = me.weapons[i].as_ref() {
                if w.ammo <= 0.0 && w.reload <= 0.0 {
                    out.reload = true;
                }
                if !ok && w.reload <= 0.0 && w.ammo <= w.stats.mag * p.reload_frac && w.ammo < w.stats.mag {
                    out.reload = true;
                }
                if w.ammo > 0.0 && w.reload <= 0.7 {
                    out.armed = true;
                }
                if allow && ok && w.ammo > 0.0 && w.reload <= 0.0 && on(a, tol_y, tol_p) {
                    out.fire[i] = true;
                }
            }
        }
        if let Some((i, w)) = laser_w {
            let full = w.stats.charge.max(0.05);
            if w.cooldown <= 1.0 {
                out.armed = true;
            }
            if !laser_charging {
                if allow && laser_in && w.cooldown <= 0.1 {
                    out.fire[i] = true;
                }
            } else {
                let ready = w.charge >= full - 0.051;
                let aligned = on(direct, tol_y, tol_p);
                let part = w.stats.damage * (w.stats.min_frac + (1.0 - w.stats.min_frac) * (w.charge / full).clamp(0.0, 1.0));
                let lethal = w.charge >= w.stats.min_charge && tg.hp > 0.0 && part >= tg.hp * 1.05;
                let release = (ready || lethal) && aligned && laser_in;
                let give_up = !laser_in && w.charge >= full + (w.stats.hold_s - 0.4).max(0.0);
                out.fire[i] = !(release || give_up);
            }
            out.charging = out.fire[i];
        }
        if let Some(i) = lightning {
            if let Some(w) = me.weapons[i].as_ref() {
                let in_range = d <= w.stats.range * 0.95 && dist(c, raised(me.pos, st.height * 0.5)) <= w.stats.range * 0.95;
                if in_range {
                    out.armed = true;
                }
                let yaw_to = rel[1].atan2(rel[0]);
                let cone = w.stats.cone_deg.to_radians() * p.lightning_cone_frac;
                if allow && in_range && line_of_sight && angle_diff(pred[0], yaw_to).abs() <= cone && me.energy >= p.lightning_min_energy.max(w.stats.energy) {
                    out.fire[i] = true;
                }
            }
        }
        if mortar_primary {
            if let Some((i, a)) = mortar {
                if on(a, (tg.radius + 1.5).atan2(d), 0.02) {
                    out.fire[i] = true;
                }
            }
        }
        out.aim = command;
        out.target = Some(tg);
        out
    }
}
