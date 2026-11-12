//! Shared declarations and timing kernels for separate codec benchmark collections.

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
                    &$crate::fixtures::codec::legal_rows::<$game, $rows, $holes>(),
                );
            )+
        )+
    };
}

mod runner;
mod six_columns;
mod standard;
