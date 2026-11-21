use std::collections::VecDeque;

use super::math::*;
use crate::io::proto::{ActorKind, Config, Event, Obs, SoundDist, TakeFailReason, WeaponKind};
use crate::sim::world::World;

#[derive(Clone, Debug)]
pub struct Track {
    pub id: i64,
    pub kind: ActorKind,
    pub pos: V3,
    pub vel: V3,
    pub accel: V3,
    pub aim: V2,
    pub hp: f64,
    pub max_hp: f64,
    pub cubes: f64,
    pub weapons: Vec<WeaponKind>,
    pub charging: bool,
    pub charge_since: f64,
    pub last_seen: f64,
    pub damage_to_us: f64,
    pub last_damage_time: f64,
}

impl Track {
    pub fn has_weapon(&self, k: WeaponKind) -> bool {
        self.weapons.contains(&k)
    }
    pub fn pos_at(&self, t: f64) -> V3 {
        let time_step = (t - self.last_seen).clamp(0.0, 1.0);
        add(self.pos, scale([self.vel[0], self.vel[1], 0.0], time_step))
    }
}

#[derive(Clone, Debug)]
pub struct Item {
    pub id: i64,
    pub pos: V3,
    pub hp: f64,
    pub kind: WeaponKind,
    pub last_seen: f64,
    pub misses: u32,
}

#[derive(Clone, Debug)]
pub struct HeardSound {
    pub kind: String,
    pub bearing: f64,
    pub dist: SoundDist,
    pub time: f64,
}

#[derive(Clone, Debug)]
pub struct UnseenAttacker {
    pub bearing: f64,
    pub time: f64,
    pub from: V3,
    pub laser: bool,
}

pub struct Memory {
    pub tracks: Vec<Track>,
    pub crates: Vec<Item>,
    pub cubes: Vec<Item>,
    pub pickups: Vec<Item>,
    pub sounds: Vec<HeardSound>,
    pub last_hurt_time: f64,
    pub last_dealt_time: f64,
    pub unseen: Option<UnseenAttacker>,
    pub position_history: VecDeque<(f64, V3)>,
    pub blacklist: Vec<(i64, f64)>,
    pub last_time: f64,
}

fn should_see(obs: &Obs, world: &World, config: &Config, p: V3) -> bool {
    let me = &obs.me;
    let eye = raised(me.pos, me.stats.eye_height);
    let d = dist(eye, p);
    let near = config.vision.near_range + me.stats.near_range_bonus - 3.0;
    let far = config.vision.range + me.stats.vision_bonus - 8.0;
    let in_range = if d <= near {
        true
    } else if d <= far {
        let yaw = (p[1] - eye[1]).atan2(p[0] - eye[0]);
        angle_diff(yaw, me.aim[0]).abs() < (config.vision.fov_deg * 0.5).to_radians() - 0.12
    } else {
        false
    };
    in_range && world.sight(eye, raised(p, 1.0))
}

fn merge_items(list: &mut Vec<Item>, seen: &[(i64, V3, f64, WeaponKind)], obs: &Obs, world: &World, cfg: &Config, check_budget: &mut i32) {
    let t = obs.t;
    for &(id, pos, hp, kind) in seen {
        if let Some(it) = list.iter_mut().find(|i| i.id == id) {
            it.pos = pos;
            it.hp = hp;
            it.kind = kind;
            it.last_seen = t;
            it.misses = 0;
        } else {
            list.push(Item { id, pos, hp, kind, last_seen: t, misses: 0 });
        }
    }
    for it in list.iter_mut() {
        if it.last_seen < t && *check_budget > 0 {
            *check_budget -= 1;
            if should_see(obs, world, cfg, it.pos) {
                it.misses += 1;
            }
        }
    }
    list.retain(|i| i.misses < 2 && i.pos.iter().all(|x| x.is_finite()));
    if list.len() > 200 {
        let cut = list.len() - 200;
        list.drain(0..cut);
    }
}

impl Memory {
    pub fn new() -> Memory {
        Memory {
            tracks: Vec::new(),
            crates: Vec::new(),
            cubes: Vec::new(),
            pickups: Vec::new(),
            sounds: Vec::new(),
            last_hurt_time: -1e9,
            last_dealt_time: -1e9,
            unseen: None,
            position_history: VecDeque::new(),
            blacklist: Vec::new(),
            last_time: -1.0,
        }
    }

    pub fn blacklisted(&self, id: i64, t: f64) -> bool {
        self.blacklist.iter().any(|&(i, until)| i == id && until > t)
    }

    pub fn blacklist_until(&mut self, id: i64, until: f64) {
        self.blacklist.retain(|&(i, _)| i != id);
        self.blacklist.push((id, until));
        if self.blacklist.len() > 64 {
            self.blacklist.remove(0);
        }
    }

    pub fn update(&mut self, obs: &Obs, world: &World, config: &Config) {
        let t = obs.t;
        let me = &obs.me;
        let time_step = if self.last_time >= 0.0 { (t - self.last_time).clamp(0.02, 2.0) } else { 0.1 };
        self.last_time = t;

        let decay = (-time_step / 4.0f64).exp();
        for tr in self.tracks.iter_mut() {
            tr.damage_to_us *= decay;
        }

        for a in &obs.mechs {
            if a.id == me.id || a.kind == ActorKind::Unknown || !a.pos.iter().all(|x| x.is_finite()) {
                continue;
            }
            let health = a.hp.unwrap_or(if a.kind == ActorKind::Npc { config.npc.hp } else { config.mech.hp });
            let max_hp = a.max_hp.unwrap_or(if a.kind == ActorKind::Npc { config.npc.hp } else { config.mech.hp });
            if let Some(tr) = self.tracks.iter_mut().find(|x| x.id == a.id) {
                let dtr = t - tr.last_seen;
                if dtr > 1e-6 && dtr <= 0.25 {
                    let na = scale(sub(a.vel, tr.vel), 1.0 / dtr);
                    tr.accel = add(scale(tr.accel, 0.5), scale(na, 0.5));
                } else {
                    tr.accel = [0.0; 3];
                }
                if a.charging && !tr.charging {
                    tr.charge_since = t;
                }
                tr.kind = a.kind;
                tr.pos = a.pos;
                tr.vel = a.vel;
                tr.aim = a.aim;
                tr.hp = health;
                tr.max_hp = max_hp;
                tr.cubes = a.cubes;
                tr.weapons = a.weapons.clone();
                tr.charging = a.charging;
                tr.last_seen = t;
            } else {
                self.tracks.push(Track {
                    id: a.id,
                    kind: a.kind,
                    pos: a.pos,
                    vel: a.vel,
                    accel: [0.0; 3],
                    aim: a.aim,
                    hp: health,
                    max_hp,
                    cubes: a.cubes,
                    weapons: a.weapons.clone(),
                    charging: a.charging,
                    charge_since: t,
                    last_seen: t,
                    damage_to_us: 0.0,
                    last_damage_time: -1e9,
                });
            }
        }

        for ev in &obs.events {
            match ev {
                Event::Damaged(d) => {
                    if d.weapon == "zone" || d.by.is_none() {
                        continue;
                    }
                    self.last_hurt_time = t;
                    let by = d.by.unwrap_or(-1);
                    let mut fresh = false;
                    if let Some(tr) = self.tracks.iter_mut().find(|x| x.id == by) {
                        tr.damage_to_us += d.amount.max(0.0);
                        tr.last_damage_time = t;
                        fresh = t - tr.last_seen < 0.3;
                    }
                    if !fresh {
                        if let Some(b) = d.bearing {
                            if b.is_finite() {
                                self.unseen = Some(UnseenAttacker { bearing: b, time: t, from: me.pos, laser: d.weapon == "laser" });
                            }
                        }
                    }
                }
                Event::Hit(h) => {
                    self.last_dealt_time = t;
                    if let Some(c) = self.crates.iter_mut().find(|c| c.id == h.target) {
                        c.hp -= h.amount;
                    }
                }
                Event::Kill(k) => {
                    self.tracks.retain(|x| x.id != k.target);
                }
                Event::Took(k) => {
                    self.pickups.retain(|x| x.id != k.id);
                }
                Event::TakeFailed(f) => match f.reason {
                    TakeFailReason::Gone => self.pickups.retain(|x| x.id != f.id),
                    TakeFailReason::Range => {}
                    _ => self.blacklist_until(f.id, t + 8.0),
                },
                _ => {}
            }
        }

        self.tracks.retain(|x| t - x.last_seen < 25.0 && !(x.kind == ActorKind::Npc && x.hp <= 0.0));
        self.tracks.sort_by_key(|x| x.id);

        let mut budget = 60i32;
        let crates: Vec<_> = obs.crates.iter().map(|c| (c.id, c.pos, c.hp.unwrap_or(config.crates.hp), WeaponKind::Unknown)).collect();
        merge_items(&mut self.crates, &crates, obs, world, config, &mut budget);
        let cubes: Vec<_> = obs.cubes.iter().map(|c| (c.id, c.pos, 0.0, WeaponKind::Unknown)).collect();
        merge_items(&mut self.cubes, &cubes, obs, world, config, &mut budget);
        let picks: Vec<_> = obs.pickups.iter().map(|c| (c.id, c.pos, 0.0, c.kind)).collect();
        merge_items(&mut self.pickups, &picks, obs, world, config, &mut budget);

        for s in &obs.sounds {
            if s.bearing.is_finite() {
                self.sounds.push(HeardSound { kind: s.kind.clone(), bearing: s.bearing, dist: s.dist, time: t - s.age.max(0.0) });
            }
        }
        self.sounds.retain(|s| t - s.time < 6.0);
        if self.sounds.len() > 64 {
            let cut = self.sounds.len() - 64;
            self.sounds.drain(0..cut);
        }
        if let Some(u) = &self.unseen {
            if t - u.time > 6.0 {
                self.unseen = None;
            }
        }
        self.blacklist.retain(|&(_, until)| until > t);

        self.position_history.push_back((t, me.pos));
        while self.position_history.front().map(|h| t - h.0 > 3.0).unwrap_or(false) {
            self.position_history.pop_front();
        }
    }

    /// Displacement over the last w seconds, or None when history is shorter.
    pub fn distance_moved(&self, t: f64, w: f64) -> Option<f64> {
        let first = self.position_history.front()?;
        if t - first.0 < w * 0.9 {
            return None;
        }
        let old = self.position_history.iter().find(|h| t - h.0 <= w)?;
        let current = self.position_history.back()?;
        Some(dist(old.1, current.1))
    }
}
