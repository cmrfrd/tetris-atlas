//! Declare and register benchmark callbacks together.

pub struct Benchmark {
    pub(crate) name: &'static str,
    pub run: fn(&mut criterion::Criterion),
}

inventory::collect!(Benchmark);

pub fn registered() -> Vec<&'static Benchmark> {
    let mut benchmarks: Vec<_> = inventory::iter::<Benchmark>.into_iter().collect();
    benchmarks.sort_unstable_by_key(|benchmark| benchmark.name);
    assert!(
        benchmarks
            .windows(2)
            .all(|pair| pair[0].name != pair[1].name),
        "duplicate benchmark declaration"
    );
    benchmarks
}

// Attributes apply to the callback and its registration as one declaration.
macro_rules! tetris_bench {
    ($(#[$meta:meta])* fn $name:ident($context:ident: $context_type:ty) $body:block) => {
        $(#[$meta])*
        const _: () = {
            fn $name($context: $context_type) $body

            ::inventory::submit! {
                $crate::registry::Benchmark {
                    name: concat!(module_path!(), "::", stringify!($name)),
                    run: $name,
                }
            }
        };
    };
}

#[cfg(test)]
mod tests {
    static REGISTRATION_PROBE_CALLS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    tetris_bench! {
        #[cfg(all())]
        fn registration_probe(_c: &mut criterion::Criterion) {
            REGISTRATION_PROBE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    tetris_bench! {
        #[cfg(any())]
        fn disabled_probe(_c: &mut criterion::Criterion) {
            compile_error!("cfg-disabled benchmark body must not be compiled");
        }
    }

    tetris_bench! {
        #[cfg_attr(all(), cfg(any()))]
        fn cfg_attr_disabled_probe(_c: &mut criterion::Criterion) {
            compile_error!("cfg_attr-disabled benchmark body must not be compiled");
        }
    }

    #[test]
    fn declarations_register_their_callbacks() {
        let benchmarks = super::registered();
        assert!(
            benchmarks
                .windows(2)
                .all(|pair| pair[0].name < pair[1].name)
        );
        assert!(
            benchmarks
                .iter()
                .all(|benchmark| !benchmark.name.ends_with("disabled_probe"))
        );

        let probe = benchmarks
            .iter()
            .find(|benchmark| benchmark.name == concat!(module_path!(), "::registration_probe"))
            .expect("macro declaration must register its callback");
        (probe.run)(&mut criterion::Criterion::default());
        assert_eq!(
            REGISTRATION_PROBE_CALLS.load(std::sync::atomic::Ordering::Relaxed),
            1
        );
    }
}
