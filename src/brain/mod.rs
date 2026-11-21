mod aim;
mod math;
mod memory;
mod params;
mod plan;
mod roof;
mod strategy;
#[cfg(test)]
mod tests;

pub use params::Params;

use std::sync::Arc;

use crate::io::proto::{self, Action, MechCfg, Obs, Offer, SelfObs, WeaponKind};
use crate::io::runtime::Budget;
use crate::sim::nav;
use crate::{Pilot, Shared};
use math::*;
use memory::Memory;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mode {
    Idle,
    Zone,
    Fight,
    Retreat,
    Loot,
    Drop,
    Hunt,
    Roof,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Goal {
    None,
    Nav { pos: V3, arrive: f64 },
    Field { center: V2, arrive: f64 },
    Range { pos: V3, pref: f64 },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Target {
    None,
    Mech(i64),
    Npc(i64),
    Crate(i64),
}

pub struct Brain {
    pub(crate) config: proto::Config,
    pub(crate) params: Arc<Params>,
    pub(crate) shared: Arc<Shared>,
    pub(crate) rng: Rng,
    pub(crate) memory: Memory,
    pub(crate) mode: Mode,
    pub(crate) goal: Goal,
    pub(crate) goal_id: i64,
    pub(crate) pursuit: (i64, f64),
    pub(crate) target: Target,
    pub(crate) allow_crates: bool,
    pub(crate) field: Option<(V3, Arc<nav::Field>, f64)>,
    pub(crate) zone_field: Option<(V2, f64, Arc<nav::Field>)>,
    pub(crate) goal_unreachable: bool,
    pub(crate) cover: Option<(V3, f64)>,
    pub(crate) last_dir: V2,
    pub(crate) juke_side: f64,
    pub(crate) juke_next: f64,
    pub(crate) unstick_until: f64,
    pub(crate) unstick_dir: V2,
    pub(crate) last_target_time: f64,
    pub(crate) zone_phase_started: bool,
    pub(crate) unstick_count: u32,
    pub(crate) unstick_level: u32,
    pub(crate) commanded_since: f64,
    pub(crate) wants_since: f64,
    pub(crate) moving_since: f64,
    pub(crate) act_started: std::time::Instant,
    pub(crate) sight_lost: (i64, f64),
    pub(crate) jet: Option<(V3, f64)>,
    pub(crate) roof: Option<Arc<roof::RoofMap>>,
    pub(crate) roof_choice: Option<usize>,
    pub(crate) roof_choice_time: f64,
    pub(crate) roof_current: Option<usize>,
    pub(crate) roof_rest: bool,
    pub(crate) roof_hold: bool,
    pub(crate) roof_jet: bool,
    pub(crate) roof_ban: Vec<(usize, f64)>,
    pub(crate) roof_step: Option<(i64, Option<V3>, f64)>,
    pub(crate) roof_first_time: f64,
    pub(crate) kills: u32,
    #[cfg(test)]
    pub(crate) debug_line: String,
}

impl Brain {
    pub fn new(hello: &proto::Hello, shared: Arc<Shared>) -> Brain {
        Brain::with_params(hello, shared, Params::load())
    }

    pub fn with_params(hello: &proto::Hello, shared: Arc<Shared>, params: Params) -> Brain {
        let mut rng = Rng::new(hello.seed as u64 ^ ((hello.slot as u64) << 40));
        let side = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
        let roof = if params.roof_mode > 0.0 {
            let (mn, ms) = (params.roof_min_nodes.max(1.0) as usize, params.roof_min_setback * 0.5);
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Arc::new(roof::RoofMap::build(&shared.world, &shared.nav, mn, ms, hello.config.mech.height)))).ok()
        } else {
            None
        };
        Brain {
            config: hello.config.clone(),
            params: Arc::new(params),
            shared,
            rng,
            memory: Memory::new(),
            mode: Mode::Idle,
            goal: Goal::None,
            goal_id: -1,
            pursuit: (-1, 0.0),
            target: Target::None,
            allow_crates: false,
            field: None,
            zone_field: None,
            goal_unreachable: false,
            cover: None,
            last_dir: [0.0, 0.0],
            juke_side: side,
            juke_next: 0.0,
            unstick_until: -1.0,
            unstick_dir: [0.0, 0.0],
            last_target_time: -1e9,
            zone_phase_started: false,
            unstick_count: 0,
            unstick_level: 0,
            commanded_since: 0.0,
            wants_since: 0.0,
            moving_since: 0.0,
            act_started: std::time::Instant::now(),
            sight_lost: (-1, 0.0),
            jet: None,
            roof,
            roof_choice: None,
            roof_choice_time: -1e9,
            roof_current: None,
            roof_rest: false,
            roof_hold: false,
            roof_jet: false,
            roof_ban: Vec::new(),
            roof_step: None,
            roof_first_time: -1.0,
            kills: 0,
            #[cfg(test)]
            debug_line: String::new(),
        }
    }

    pub fn act(&mut self, obs: &Obs, budget: &Budget) -> Action {
        self.act_started = std::time::Instant::now();
        let mut act = Action::new(obs.tick);
        let shared = self.shared.clone();
        let world = &shared.world;
        let nav = &shared.nav;
        self.memory.update(obs, world, &self.config);
        if let Some(of) = &obs.offer {
            act.pick = Some(self.pick_upgrade(of));
        }
        self.decide(obs, world, nav);
        let fc = self.fire_control(obs, world);
        let move_dir = self.plan_move(obs, world, nav, budget, &fc);
        self.loot_action(obs, &mut act);
        act.move_ = Some(move_dir.dir);
        act.boost = Some(move_dir.boost);
        act.jump = Some(move_dir.jump);
        act.aim = Some(fc.aim);
        act.fire = Some(fc.fire);
        act.reload = Some(fc.reload);
        if self.params.halt_after >= 0.0 && obs.t >= self.params.halt_after {
            act.move_ = Some([0.0, 0.0]);
            act.boost = Some(false);
            act.jump = Some(false);
            act.fire = Some([false, false]);
            act.reload = Some(false);
            act.take = None;
            act.drop = None;
        }
        sanitize_action(&mut act, &obs.me.stats);
        act
    }

    pub(crate) fn is_holding(&self, t: f64) -> bool {
        self.params.hold_until >= 0.0 && t < self.params.hold_until
    }

    /// Short state line for logs and tools.
    pub fn state(&self) -> String {
        format!("{:?} {:?} {:?} tracks={} items={}/{}/{} roof={:?}/{:?}/{} first={:.1}", self.mode, self.goal, self.target, self.memory.tracks.len(), self.memory.crates.len(), self.memory.cubes.len(), self.memory.pickups.len(), self.roof_choice, self.roof_current, self.roof_rest, self.roof_first_time)
    }

    pub(crate) fn elapsed_ms(&self) -> f64 {
        self.act_started.elapsed().as_secs_f64() * 1000.0
    }

    pub fn unsticks(&self) -> u32 {
        self.unstick_count
    }

    fn pick_upgrade(&self, of: &Offer) -> i64 {
        for id in &self.params.upgrade_priority {
            if let Some(i) = of.choices.iter().take(3).position(|c| &c.id == id) {
                return i as i64;
            }
        }
        0
    }

    pub(crate) fn weapon_preference(&self, k: WeaponKind) -> f64 {
        match k {
            WeaponKind::Gun => self.params.pref_gun,
            WeaponKind::Laser => self.params.pref_laser,
            WeaponKind::Lightning => self.params.pref_lightning,
            WeaponKind::Mortar => self.params.pref_mortar,
            WeaponKind::Unknown => -1.0,
        }
    }

    /// None: not wanted. Some(None): goes into an empty slot. Some(Some(s)): swap out slot s.
    pub(crate) fn wanted_slot(&self, me: &SelfObs, kind: WeaponKind) -> Option<Option<i64>> {
        if kind == WeaponKind::Unknown || me.weapons.iter().flatten().any(|w| w.kind == kind) {
            return None;
        }
        if me.weapons.iter().any(|w| w.is_none()) {
            return Some(None);
        }
        let mut worst: Option<(f64, usize)> = None;
        for (i, w) in me.weapons.iter().enumerate() {
            let Some(w) = w else { continue };
            if w.kind == WeaponKind::Laser && w.charge > 0.0 {
                continue;
            }
            let pv = self.weapon_preference(w.kind);
            if worst.map_or(true, |x| pv < x.0) {
                worst = Some((pv, i));
            }
        }
        match worst {
            Some((pv, i)) if pv + 0.5 < self.weapon_preference(kind) => Some(Some(i as i64)),
            _ => None,
        }
    }

    fn loot_action(&mut self, obs: &Obs, act: &mut Action) {
        let me = &obs.me;
        let reach = (self.config.loot.pickup_radius - 0.3).max(0.5);
        let mut best: Option<(f64, i64, Option<i64>)> = None;
        for pk in &obs.pickups {
            if horiz_dist(pk.pos, me.pos) > reach || (pk.pos[2] - me.pos[2]).abs() > 3.7 {
                continue;
            }
            if self.memory.blacklisted(pk.id, obs.t) {
                continue;
            }
            if let Some(slot) = self.wanted_slot(me, pk.kind) {
                let v = self.weapon_preference(pk.kind);
                if best.map_or(true, |b| v > b.0) {
                    best = Some((v, pk.id, slot));
                }
            }
        }
        if let Some((_, id, slot)) = best {
            act.take = Some(id);
            act.drop = slot;
        }
    }
}

fn sanitize_action(act: &mut Action, mech: &MechCfg) {
    let mut m = act.move_.unwrap_or([0.0, 0.0]);
    if !m[0].is_finite() || !m[1].is_finite() {
        m = [0.0, 0.0];
    }
    let l = m[0].hypot(m[1]);
    if l > 1.0 {
        m = [m[0] / l * 0.999_999, m[1] / l * 0.999_999];
    }
    act.move_ = Some(m);
    let a = act.aim.unwrap_or([0.0, 0.0]);
    let (lo, hi) = if mech.min_pitch < mech.max_pitch { (mech.min_pitch, mech.max_pitch) } else { (-0.7, 1.3) };
    act.aim = Some([normalize_angle(a[0]), finite_or(a[1], 0.0).clamp(lo, hi)]);
    if act.fire.is_none() {
        act.fire = Some([false, false]);
    }
    if let Some(pk) = act.pick {
        act.pick = Some(pk.clamp(0, 2));
    }
}

impl Pilot for Brain {
    fn act(&mut self, obs: &Obs) -> Action {
        Brain::act(self, obs, &Budget::unlimited())
    }
}
