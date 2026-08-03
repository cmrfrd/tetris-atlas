/// Genetic-operator tuning. Defaults favor deep, sizeable, and numerous trees,
/// and cull any tree that ever outputs an invalid placement (`min_valid_rate` 1.0).
#[derive(Clone, Copy, Debug)]
pub struct EvolveConfig {
    /// Depth used when RANDOMLY GENERATING trees (population init, tree-add
    /// mutation). Kept small: random tree generation is a supercritical
    /// branching process (expected size ~1.5^depth), so trees start small and
    /// GROW via mutation instead. Deepening happens through evolution.
    pub init_tree_depth: u32,
    /// Structural cap on decision-tree depth: grow mutations stop once a tree
    /// reaches this depth (deep trees allowed, unbounded bloat not).
    pub max_tree_depth: u32,
    /// Structural cap on boolean-condition depth (growth). Random GENERATION
    /// is additionally capped at `COND_GEN_DEPTH_CAP` (4) — generated
    /// conditions are a supercritical branching process, so deep generation
    /// bloats clone/eval cost with unselected random structure.
    pub max_cond_depth: u32,
    pub leaf_mutation_prob: f64,
    pub cond_mutation_prob: f64,
    pub grow_prob: f64,
    pub prune_prob: f64,
    pub tree_add_prob: f64,
    pub tree_remove_prob: f64,
    pub max_trees: usize,
    /// Regrowth floor: after the autokill, offspring are topped back up with
    /// fresh piece-gated trees (100% valid by construction) to at least this
    /// many. The counterweight to the cull — keeps the vote a genuine
    /// many-tree accumulation instead of ratcheting down to one survivor.
    /// `0` disables regrowth.
    pub min_trees: usize,
    /// Autokill threshold: each generation, trees whose valid-output rate over a
    /// sample game is below this are culled. Default `1.0` = cull any tree that
    /// ever outputs an invalid placement. `0.0` disables culling.
    pub min_valid_rate: f64,
    /// Pieces per sample game used to measure per-tree validity for the cull.
    pub cull_sample_pieces: u64,
}

impl Default for EvolveConfig {
    fn default() -> Self {
        EvolveConfig {
            init_tree_depth: 6,
            max_tree_depth: 24,
            max_cond_depth: 6,
            leaf_mutation_prob: 0.25,
            cond_mutation_prob: 0.25,
            grow_prob: 0.15,
            prune_prob: 0.10,
            tree_add_prob: 0.10,
            tree_remove_prob: 0.03,
            max_trees: 256,
            min_trees: 32,
            min_valid_rate: 1.0,
            cull_sample_pieces: 300,
        }
    }
}

/// Training-loop tuning.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub population: usize,
    pub generations: usize,
    pub elite: usize,
    pub tournament_k: usize,
    pub seeds: usize,
    /// Eval-seed rotation window: the seed set is FROZEN for this many
    /// generations, then rotated. A frozen block gives selection a stationary
    /// objective to climb (rotating every generation made champion acceptance
    /// a race against common-mode seed noise); rotation across blocks still
    /// prevents overfitting one sequence set.
    pub seed_block: usize,
    pub start_cap: u64,
    pub init_trees: usize,
    /// Print a progress line every N generations (1 = every generation).
    pub log_every: usize,
    /// Save a champion snapshot (`<run>_gen_<n>.json`) every N generations.
    /// `None` (the default) disables snapshots — the synchronous write causes
    /// a visible pause, and the champion JSON is saved on every improvement
    /// anyway.
    pub snapshot_every: Option<usize>,
    pub evolve: EvolveConfig,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            population: 64,
            generations: 1000,
            elite: 8,
            tournament_k: 5,
            seeds: 8,
            seed_block: 20,
            start_cap: 200,
            init_trees: 64,
            log_every: 1,
            snapshot_every: None,
            evolve: EvolveConfig::default(),
        }
    }
}
