//! Shared declarations and fixtures for separate codec benchmark collections.

/// Instantiates each selected row/hole pair using the same public-API timing kernels.
macro_rules! codec_grid {
    ($c:ident, $game:ty, $collection:literal, $prefix:literal;
        $($rows:literal => [$($holes:literal),+ $(,)?]),+ $(,)?) => {
        $(
            $(
                super::runner::bench_config::<$game, $rows, $holes>(
                    $c,
                    $collection,
                    concat!($prefix, "_", stringify!($rows), "_", stringify!($holes)),
                    &super::fixtures::legal_rows::<$game, $rows, $holes>(),
                );
            )+
        )+
    };
}

mod fixtures;
mod runner;
mod six_columns;
mod standard;

use runner::SAMPLES;
