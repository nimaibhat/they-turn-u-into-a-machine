use super::math::*;
use super::memory::Track;
use super::{Brain, Goal, Mode, Target};
use crate::sim::nav::Nav;
use crate::io::proto::{ActorKind, Obs, SelfObs, WeaponKind, ZoneObs};
use crate::sim::world::World;

pub(crate) struct ZoneInfo {
    pub urgent: bool,
    pub safe: V2,
    pub center: V2,
    pub radius: f64,
    pub closing: bool,
}

/// Circle at time tau, given the zone reported at time now.
pub(crate) fn zone_at(z: &ZoneObs, now: f64, tau: f64, phase_start: bool) -> (V2, f64) {
    let t0 = if phase_start { z.shrink_start } else { z.shrink_start.max(now) };
    let span = z.shrink_end - t0;
    if tau <= t0 || !(span > 1e-6) {
        return (z.center, z.radius);
    }
    let k = ((tau - t0) / span).clamp(0.0, 1.0);
    (
        [z.center[0] + (z.next_center[0] - z.center[0]) * k, z.center[1] + (z.next_center[1] - z.center[1]) * k],
        z.radius + (z.next_radius - z.radius) * k,
    )
}

struct LootPick {
    id: i64,
    pos: V3,
    quick: bool,
}

fn ground(world: &World, p: V2) -> V3 {
    [p[0], p[1], finite_or(world.terrain_z(p[0], p[1]), 0.0)]
}

impl Brain {
    fn safe_radius(&self, z: &ZoneObs) -> f64 {
        let p = &self.params;
        if z.next_radius <= p.final_radius {
            0.0
        } else {
            (z.next_radius - p.zone_margin.min(z.next_radius * 0.5) - p.zone_extra).max(0.0)
        }
    }

    fn seed_radius(&self, z: &ZoneObs) -> f64 {
        self.safe_radius(z).min(z.next_radius * self.params.idle_frac)
    }

    fn ensure_zone_field(&mut self, obs: &Obs, world: &World, nav: &Nav) {
        let z = &obs.zone;
        if let Some((c, r, _)) = &self.zone_field {
            if (c[0] - z.next_center[0]).hypot(c[1] - z.next_center[1]) < 0.5 && (r - z.next_radius).abs() < 0.5 {
                return;
            }
        }
        if !(z.next_center[0].is_finite() && z.next_center[1].is_finite() && z.next_radius.is_finite()) {
            return;
        }
        let safe_r = self.seed_radius(z);
        let mut seeds: Vec<(V3, f64)> = Vec::new();
        let c = z.next_center;
        seeds.push((ground(world, c), 0.0));
        if safe_r >= 8.0 {
            for (fr, n) in [(0.35, 8), (0.7, 12), (1.0, 20)] {
                for k in 0..n {
                    let a = k as f64 * 2.0 * std::f64::consts::PI / n as f64;
                    seeds.push((ground(world, [c[0] + safe_r * fr * a.cos(), c[1] + safe_r * fr * a.sin()]), 0.0));
                }
            }
        } else {
            for (r, n) in [(4.0, 6), (9.0, 8), (15.0, 10), (23.0, 12), (33.0, 14)] {
                for k in 0..n {
                    let a = k as f64 * 2.0 * std::f64::consts::PI / n as f64;
                    seeds.push((ground(world, [c[0] + r * a.cos(), c[1] + r * a.sin()]), 4.0 * r));
                }
            }
        }
        let jet = if safe_r < 8.0 { obs.me.max_energy * 0.8 } else { -1.0 };
        let f = nav.field_multi(&seeds, jet);
        self.zone_field = Some((c, z.next_radius, std::sync::Arc::new(f)));
    }

    pub(crate) fn zone_info(&mut self, obs: &Obs, world: &World, nav: &Nav) -> ZoneInfo {
        self.ensure_zone_field(obs, world, nav);
        let p = self.params.clone();
        let z = &obs.zone;
        let t = obs.t;
        let pos = obs.me.pos;
        let (c_a, r_a) = zone_at(z, t, t, false);
        if !obs.me.in_zone && horiz_dist_to(pos, c_a) < r_a - 1.0 && z.radius > 0.0 {
            self.zone_phase_started = true;
        }
        let (cur_c, cur_r) = zone_at(z, t, t, self.zone_phase_started);
        let dist_next = horiz_dist_to(pos, z.next_center);
        let is_final = z.next_radius <= p.final_radius;
        let margin = p.zone_margin.min(z.next_radius * 0.5);
        let inside_next = dist_next <= z.next_radius - margin;
        let safe_r = self.safe_radius(z);
        let _ = is_final;
        let safe = if dist_next > safe_r && dist_next > 1e-6 {
            [z.next_center[0] + (pos[0] - z.next_center[0]) / dist_next * safe_r, z.next_center[1] + (pos[1] - z.next_center[1]) / dist_next * safe_r]
        } else {
            [pos[0], pos[1]]
        };
        let d_need = (dist_next - (z.next_radius - margin)).max(0.0);
        let out_now = horiz_dist_to(pos, cur_c) - cur_r;
        let mut t_hit = f64::INFINITY;
        if z.shrink_end > t {
            let mut tau = t;
            while tau <= z.shrink_end + 0.5 {
                let (c, r) = zone_at(z, t, tau, self.zone_phase_started);
                if horiz_dist_to(pos, c) > r - p.zone_edge {
                    t_hit = tau;
                    break;
                }
                tau += 0.5;
            }
        }
        let walk = obs.me.stats.walk_speed.max(1.0);
        let d_path = self.zone_field.as_ref().and_then(|(_, _, f)| nav.dist(f, pos)).filter(|d| d.is_finite());
        let seed_r = self.seed_radius(z);
        let straight = (dist_next - seed_r).max(0.0);
        let need_s = match d_path {
            Some(d) => (d - (z.next_radius - margin - seed_r).max(0.0)).max(d_need) * 1.15 / walk,
            None => d_need * p.path_factor / walk,
        };
        let detour_s = d_path.map_or(0.0, |d| (d - straight).max(0.0) / walk);
        let urgent = !obs.me.in_zone
            || out_now > -p.zone_edge
            || (t_hit - t) < p.zone_lead_s + detour_s
            || (!inside_next && (z.shrink_end - t) < need_s + p.zone_slack_s);
        let closing = !inside_next && (z.shrink_start - t) < 25.0;
        ZoneInfo { urgent, safe, center: cur_c, radius: cur_r, closing }
    }

    pub(crate) fn decide(&mut self, obs: &Obs, world: &World, nav: &Nav) {
        let p = self.params.clone();
        let me = &obs.me;
        let t = obs.t;
        let pos = me.pos;
        let hp_frac = clamped_ratio(me.hp, me.max_hp);
        let zi = self.zone_info(obs, world, nav);
        let muzzle = raised(pos, me.stats.muzzle_height);

        if self.goal_unreachable {
            if self.goal_id >= 0 {
                self.memory.blacklist_until(self.goal_id, t + p.blacklist_s);
            }
            self.goal_unreachable = false;
        }
        if self.goal_id >= 0 {
            if self.pursuit.0 != self.goal_id {
                self.pursuit = (self.goal_id, t);
            } else if t - self.pursuit.1 > p.pursuit_max_s {
                self.memory.blacklist_until(self.goal_id, t + p.blacklist_s);
                self.pursuit = (-1, t);
            }
        }

        let mut n_att = 0;
        let mut near_mech = f64::INFINITY;
        for tr in &self.memory.tracks {
            if tr.kind != ActorKind::Mech {
                continue;
            }
            if t - tr.last_damage_time < 2.0 {
                n_att += 1;
            }
            if t - tr.last_seen < 1.0 {
                near_mech = near_mech.min(dist(tr.pos, pos));
            }
        }
        let threat_near = near_mech < 200.0 || t - self.memory.last_hurt_time < 3.0;
        let laser_near = self.memory.tracks.iter().any(|tr| {
            tr.kind == ActorKind::Mech && t - tr.last_seen < 1.0 && tr.has_weapon(WeaponKind::Laser) && dist(tr.pos_at(t), pos) < 130.0 && world.sight(raised(tr.pos_at(t), self.config.mech.eye_height), raised(pos, me.stats.height * 0.6))
        });
        let pressured = n_att >= 2 || (laser_near && t - self.memory.last_hurt_time < 4.0);
        let held = self.is_holding(t);
        let tgt = if held { None } else { self.choose_target(obs, &zi) };
        let mut near_mem = f64::INFINITY;
        for tr in &self.memory.tracks {
            if tr.kind == ActorKind::Mech && t - tr.last_seen < 4.0 {
                near_mem = near_mem.min(dist(tr.pos_at(t), pos));
            }
        }

        let was = self.mode == Mode::Retreat;
        let mut want_retreat = if was { hp_frac < p.retreat_exit_hp } else { (hp_frac < p.retreat_hp && threat_near) || (n_att >= 2 && hp_frac < p.retreat_multi_hp) || (pressured && hp_frac < p.retreat_press_hp) };
        if want_retreat && !was {
            if let Some(tr) = &tgt {
                if n_att <= 1 && tr.hp < me.hp * p.finish_ratio && t - tr.last_seen < 0.3 {
                    want_retreat = false;
                }
            }
        }

        if held && (n_att >= 1 || near_mech < 120.0 || near_mem < 100.0 || t - self.memory.last_hurt_time < 3.0 || (was && near_mem < 170.0)) {
            want_retreat = true;
        }
        let loot = self.choose_loot(obs, &zi);
        let drop = self.choose_drop(obs, &zi).filter(|_| !held || near_mem > 100.0);
        let z = &obs.zone;
        let endgame = held && z.next_radius <= 0.5 && z.shrink_end - t < 40.0;
        let mut mode;
        let goal;
        let mut target = Target::None;
        let mut goal_id = -1;
        if let Some(tr) = &tgt {
            target = Target::Mech(tr.id);
        }

        if endgame {
            mode = Mode::Zone;
            goal = Goal::Field { center: z.next_center, arrive: 0.8 };
        } else if zi.urgent {
            mode = Mode::Zone;
            goal = Goal::Field { center: obs.zone.next_center, arrive: 1.5 };
        } else if want_retreat {
            mode = Mode::Retreat;
            goal = self.cover_goal(obs, world, nav, &zi);
        } else if let Some(l) = loot.as_ref().filter(|l| l.quick) {
            mode = Mode::Loot;
            goal = Goal::Nav { pos: l.pos, arrive: 1.2 };
            goal_id = l.id;
        } else if let Some(d) = drop.filter(|d| tgt.is_none() || d.2 < 50.0 || hp_frac < 0.6) {
            mode = Mode::Drop;
            goal = Goal::Nav { pos: d.0, arrive: (d.1 * 0.3).max(1.0) };
        } else if let Some(tr) = &tgt {
            mode = Mode::Fight;
            let track_pos = tr.pos_at(t);
            let c = raised(track_pos, self.config.mech.height * 0.5);
            let line_of_sight = t - tr.last_seen < p.fresh_s && world.ray(muzzle, c).is_none();
            if line_of_sight || t - tr.last_damage_time < 3.0 {
                self.sight_lost = (tr.id, t);
            } else if self.sight_lost.0 != tr.id {
                self.sight_lost = (tr.id, t);
            } else if t - self.sight_lost.1 > p.no_los_drop_s {
                self.memory.blacklist_until(tr.id, t + 8.0);
            }
            let preferred = self.pref_range(me, tr, hp_frac);
            goal = if line_of_sight { Goal::Range { pos: track_pos, pref: preferred } } else { Goal::Nav { pos: track_pos, arrive: (preferred * 0.5).max(15.0) } };
        } else if let Some(l) = loot.as_ref() {
            mode = Mode::Loot;
            goal = Goal::Nav { pos: l.pos, arrive: 1.2 };
            goal_id = l.id;
        } else {
            mode = Mode::Idle;
            if let Some((tg, g, id)) = self.choose_hunt(obs, world, &zi, hp_frac, near_mech, muzzle) {
                mode = Mode::Hunt;
                target = tg;
                goal = g;
                goal_id = id;
            } else {
                goal = Goal::Field { center: obs.zone.next_center, arrive: 2.0 };
            }
        }
        self.allow_crates = matches!(mode, Mode::Hunt | Mode::Loot | Mode::Idle) && hp_frac >= p.crate_min_hp && near_mech > p.hunt_safe_range;
        if mode != Mode::Retreat {
            self.cover = None;
        }
        if goal_id != self.goal_id {
            self.pursuit = (goal_id, t);
        }
        self.mode = mode;
        self.goal = goal;
        self.goal_id = goal_id;
        self.target = target;
        self.roof_decide(obs, world, nav, &zi);
    }

    fn choose_target(&self, obs: &Obs, zi: &ZoneInfo) -> Option<Track> {
        let p = &self.params;
        let t = obs.t;
        let pos = obs.me.pos;
        let z = &obs.zone;
        let mut best: Option<(f64, &Track)> = None;
        for tr in &self.memory.tracks {
            if tr.kind != ActorKind::Mech || t - tr.last_seen > p.target_memory_s {
                continue;
            }
            let track_pos = tr.pos_at(t);
            let d = dist(track_pos, pos);
            let attacker = t - tr.last_damage_time < 5.0;
            if !attacker && self.memory.blacklisted(tr.id, t) {
                continue;
            }
            if d > p.engage_range && !attacker {
                continue;
            }
            if d > 60.0 {
                if horiz_dist_to(track_pos, zi.center) > zi.radius - 2.0 {
                    continue;
                }
                if zi.closing && horiz_dist_to(track_pos, z.next_center) > z.next_radius + p.chase_out_next {
                    continue;
                }
            }
            let mut s = 60.0 * (1.0 - d / 200.0) + 80.0 * (1.0 - clamped_ratio(tr.hp, tr.max_hp));
            if attacker {
                s += 100.0 + tr.damage_to_us * 0.1;
            }
            if self.target == Target::Mech(tr.id) {
                s += p.target_sticky;
            }
            if t - tr.last_seen < p.fresh_s {
                s += 30.0;
            }
            if best.map_or(true, |b| s > b.0) {
                best = Some((s, tr));
            }
        }
        best.map(|b| b.1.clone())
    }

    fn pref_range(&self, me: &SelfObs, track: &Track, hp_frac: f64) -> f64 {
        let p = &self.params;
        let has = |k: WeaponKind| me.weapons.iter().flatten().any(|w| w.kind == k);
        let mut preferred = p.range_gun;
        if has(WeaponKind::Laser) {
            preferred = p.range_laser;
        } else if has(WeaponKind::Mortar) && !has(WeaponKind::Gun) {
            preferred = p.range_mortar;
        }
        if has(WeaponKind::Lightning) && me.energy > 35.0 && hp_frac > p.lightning_rush_hp {
            preferred = p.range_lightning;
        } else if track.has_weapon(WeaponKind::Lightning) {
            preferred = preferred.max(p.range_vs_lightning);
        }
        preferred
    }

    fn in_play(&self, obs: &Obs, zi: &ZoneInfo, q: V3, d: f64) -> bool {
        let z = &obs.zone;
        if horiz_dist_to(q, zi.center) > zi.radius - 6.0 {
            return false;
        }
        let walk = obs.me.stats.walk_speed.max(1.0);
        let out_next = horiz_dist_to(q, z.next_center) - z.next_radius;
        if out_next > -5.0 && (z.shrink_end - obs.t) < 25.0 + d / walk + out_next.max(0.0) / walk {
            return false;
        }
        true
    }

    fn choose_loot(&self, obs: &Obs, zi: &ZoneInfo) -> Option<LootPick> {
        let p = &self.params;
        let me = &obs.me;
        let t = obs.t;
        let mut best: Option<(f64, LootPick)> = None;
        for it in &self.memory.pickups {
            if self.memory.blacklisted(it.id, t) {
                continue;
            }
            let Some(slot) = self.wanted_slot(me, it.kind) else { continue };
            let d = dist(it.pos, me.pos);
            let empty = slot.is_none();
            if d > if empty { p.loot_weapon_max } else { p.loot_swap_max } || !self.in_play(obs, zi, it.pos, d) {
                continue;
            }
            let mut s = d - if empty { 80.0 } else { 0.0 } - self.weapon_preference(it.kind) * 5.0;
            if it.id == self.goal_id {
                s -= 15.0;
            }
            if best.as_ref().map_or(true, |b| s < b.0) {
                best = Some((s, LootPick { id: it.id, pos: it.pos, quick: d < p.loot_quick && empty }));
            }
        }
        if me.cubes < self.config.cubes.stat_cap {
            for it in &self.memory.cubes {
                if self.memory.blacklisted(it.id, t) {
                    continue;
                }
                let d = dist(it.pos, me.pos);
                if d > p.loot_cube_max || !self.in_play(obs, zi, it.pos, d) {
                    continue;
                }
                let mut s = d - 20.0;
                if it.id == self.goal_id {
                    s -= 15.0;
                }
                if best.as_ref().map_or(true, |b| s < b.0) {
                    best = Some((s, LootPick { id: it.id, pos: it.pos, quick: d < p.loot_quick * 0.5 }));
                }
            }
        }
        best.map(|b| b.1)
    }

    fn choose_drop(&self, obs: &Obs, zi: &ZoneInfo) -> Option<(V3, f64, f64)> {
        let p = &self.params;
        let me = &obs.me;
        let walk = me.stats.walk_speed.max(1.0);
        let mut best: Option<(V3, f64, f64)> = None;
        for d in &obs.drops {
            if let Some(h) = d.holder {
                if h != me.id && d.progress > d.capture_s * 0.5 {
                    continue;
                }
            }
            let dd = dist(d.pos, me.pos);
            if dd > p.drop_max_dist || !d.pos.iter().all(|x| x.is_finite()) {
                continue;
            }
            if d.lands_in > dd * p.path_factor / walk + p.drop_early_s {
                continue;
            }
            if horiz_dist_to(d.pos, zi.center) > zi.radius - 4.0 {
                continue;
            }
            if best.map_or(true, |b| dd < b.2) {
                best = Some((d.pos, d.radius.max(2.0), dd));
            }
        }
        best
    }

    fn choose_hunt(&self, obs: &Obs, world: &World, zi: &ZoneInfo, hp_frac: f64, near_mech: f64, muzzle: V3) -> Option<(Target, Goal, i64)> {
        let p = &self.params;
        let me = &obs.me;
        let t = obs.t;
        if hp_frac < p.hunt_min_hp || near_mech < p.hunt_safe_range {
            return None;
        }
        let more_cubes = me.cubes < self.config.cubes.stat_cap;
        let empty_slot = me.weapons.iter().any(|w| w.is_none());
        let mut best: Option<(f64, Target, Goal, i64)> = None;
        if more_cubes {
            for tr in &self.memory.tracks {
                if tr.kind != ActorKind::Npc || self.memory.blacklisted(tr.id, t) {
                    continue;
                }
                let d = dist(tr.pos, me.pos);
                if d > p.hunt_max_dist || !self.in_play(obs, zi, tr.pos, d) {
                    continue;
                }
                let mut s = d;
                if self.goal_id == tr.id {
                    s -= 20.0;
                }
                if best.as_ref().map_or(true, |b| s < b.0) {
                    let c = raised(tr.pos, self.config.npc.height * 0.5);
                    let line_of_sight = t - tr.last_seen < p.fresh_s && world.ray(muzzle, c).is_none();
                    let g = if line_of_sight { Goal::Range { pos: tr.pos, pref: p.range_npc } } else { Goal::Nav { pos: tr.pos, arrive: p.range_npc * 0.8 } };
                    best = Some((s, Target::Npc(tr.id), g, tr.id));
                }
            }
        }
        if (more_cubes || empty_slot) && hp_frac >= p.crate_min_hp {
            for it in &self.memory.crates {
                if self.memory.blacklisted(it.id, t) || it.hp <= 0.0 {
                    continue;
                }
                let d = dist(it.pos, me.pos);
                if d > p.crate_max_dist || !self.in_play(obs, zi, it.pos, d) {
                    continue;
                }
                let mut s = d - 20.0;
                if self.goal_id == it.id {
                    s -= 20.0;
                }
                if best.as_ref().map_or(true, |b| s < b.0) {
                    let c = raised(it.pos, self.config.crates.height * 0.5);
                    let los = world.ray(muzzle, c).is_none();
                    let g = if los && d < 120.0 { Goal::Range { pos: it.pos, pref: p.range_crate } } else { Goal::Nav { pos: it.pos, arrive: p.range_crate } };
                    best = Some((s, Target::Crate(it.id), g, it.id));
                }
            }
        }
        best.map(|b| (b.1, b.2, b.3))
    }

    pub(crate) fn threat_eyes(&self, obs: &Obs) -> Vec<V3> {
        let t = obs.t;
        let pos = obs.me.pos;
        let mut v: Vec<(f64, V3)> = Vec::new();
        for tr in &self.memory.tracks {
            if tr.kind != ActorKind::Mech || t - tr.last_seen > 3.0 {
                continue;
            }
            let track_pos = tr.pos_at(t);
            let s = 1.0 + self.config.cubes.size_per_cube * tr.cubes.max(0.0);
            v.push((dist(track_pos, pos), raised(track_pos, self.config.mech.eye_height * s)));
        }
        if let Some(u) = &self.memory.unseen {
            if t - u.time < 4.0 {
                let q = [u.from[0] + u.bearing.cos() * self.params.ghost_range, u.from[1] + u.bearing.sin() * self.params.ghost_range, u.from[2] + self.config.mech.eye_height];
                v.push((self.params.ghost_range, q));
            }
        }
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        v.truncate(4);
        v.into_iter().map(|x| x.1).collect()
    }

    fn cover_goal(&mut self, obs: &Obs, world: &World, nav: &Nav, zi: &ZoneInfo) -> Goal {
        let p = self.params.clone();
        let t = obs.t;
        let me = &obs.me;
        let pos = me.pos;
        if let Some((c, tc)) = self.cover {
            if t - tc < p.cover_period_s {
                return Goal::Nav { pos: c, arrive: 1.5 };
            }
        }
        let eyes = self.threat_eyes(obs);
        if eyes.is_empty() {
            self.cover = None;
            return Goal::None;
        }
        let h = me.stats.height.max(1.0);
        let hidden = |q: V3| -> bool { eyes.iter().all(|e| !world.sight(*e, raised(q, h * 0.5)) && !world.sight(*e, raised(q, h * 0.92))) };
        let z = &obs.zone;
        let mut best: Option<(f64, V3)> = None;
        let consider = |q: V3, bonus: f64, best: &mut Option<(f64, V3)>| {
            if horiz_dist_to(q, zi.center) > zi.radius - 12.0 {
                return;
            }
            if !hidden(q) {
                return;
            }
            let out_next = (horiz_dist_to(q, z.next_center) - z.next_radius + 10.0).max(0.0);
            let near_threat = eyes.iter().map(|e| horiz_dist(*e, q)).fold(f64::INFINITY, f64::min);
            let s = dist(q, pos) + 0.6 * out_next - 0.15 * near_threat.min(150.0) - bonus;
            if best.map_or(true, |b| s < b.0) {
                *best = Some((s, q));
            }
        };
        if let Some((c, _)) = self.cover {
            consider(c, 10.0, &mut best);
        }
        consider(pos, 6.0, &mut best);
        let t_act = self.act_started;
        for &r in &[12.0, 24.0, 40.0, 60.0] {
            if t_act.elapsed().as_secs_f64() > 0.010 {
                break;
            }
            for k in 0..12 {
                let a = k as f64 * std::f64::consts::PI / 6.0 + r * 0.1;
                let x = pos[0] + r * a.cos();
                let y = pos[1] + r * a.sin();
                let q0 = [x, y, finite_or(world.floor_below(x, y, pos[2] + 3.0), pos[2])];
                let Some(q) = nav.snap(q0) else { continue };
                if dist(q, q0) > 8.0 || !q.iter().all(|v| v.is_finite()) {
                    continue;
                }
                consider(q, 0.0, &mut best);
            }
        }
        let q = match best {
            Some((_, q)) => q,
            None => {
                let e = eyes[0];
                let away = unit_xy(pos[0] - e[0], pos[1] - e[1]);
                let tc = unit_xy(zi.safe[0] - pos[0], zi.safe[1] - pos[1]);
                let d = unit_xy(away[0] + 0.4 * tc[0], away[1] + 0.4 * tc[1]);
                let x = pos[0] + d[0] * p.flee_dist;
                let y = pos[1] + d[1] * p.flee_dist;
                let q0 = [x, y, finite_or(world.floor_below(x, y, pos[2] + 3.0), pos[2])];
                nav.snap(q0).filter(|q| q.iter().all(|v| v.is_finite())).unwrap_or(q0)
            }
        };
        self.cover = Some((q, t));
        Goal::Nav { pos: q, arrive: 1.5 }
    }
}
