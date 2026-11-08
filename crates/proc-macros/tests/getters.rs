#![feature(generic_const_exprs)]

use proc_macros::Getters;

#[derive(Getters)]
struct Record<'a, T: Copy, const N: usize = 2>
where
    [(); N + 1]:,
{
    #[getter(copy)]
    value: T,
    #[getter]
    bytes: [u8; N],
    #[getter(copy, name = "label")]
    text: &'a str,
    unexposed: bool,
}

const RECORD: Record<'static, u8> = Record {
    value: 7,
    bytes: [3, 4],
    text: "example",
    unexposed: true,
};
const VALUE: u8 = RECORD.value();
const BYTES: &[u8; 2] = RECORD.bytes();
const LABEL: &str = RECORD.label();

#[test]
fn getters_support_const_evaluation_and_generic_bounds() {
    assert_eq!(VALUE, 7);
    assert_eq!(BYTES, &[3, 4]);
    assert_eq!(LABEL, "example");
    assert!(std::hint::black_box(RECORD.unexposed));
}

#[test]
fn borrowed_getter_references_the_original_non_copy_field() {
    #[derive(Getters)]
    struct Owned {
        #[getter]
        text: String,
    }
    let owned = Owned {
        text: "example".to_owned(),
    };
    assert!(std::ptr::eq(owned.text(), &owned.text));
}

#[test]
fn tuple_getter_supports_custom_names_and_array_lengths() {
    #[derive(Getters)]
    struct Limbs<const N: usize>(#[getter(copy, name = "as_limbs")] [u32; N]);
    const VALUES: [u32; 3] = Limbs([1, 2, 3]).as_limbs();
    assert_eq!(VALUES, [1, 2, 3]);
}
