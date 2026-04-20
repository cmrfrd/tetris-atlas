#![allow(clippy::too_many_arguments)]

use crate::tensorboard_generated::tensorboard::summary::value;
use std::io::Cursor;

use crate::tensorboard_generated::tensorboard::{
    HistogramProto, Summary,
    summary::{Image as SummaryImage, Value as SummaryValue},
};

use image::{DynamicImage, ImageFormat, RgbImage};

pub fn scalar(name: &str, scalar_value: f32) -> Summary {
    let value = SummaryValue {
        tag: name.to_string(),
        value: Some(value::Value::SimpleValue(scalar_value)),
        ..SummaryValue::default()
    };

    Summary { value: vec![value] }
}

pub fn histogram_raw(
    name: &str,
    min: f64,
    max: f64,
    num: f64,
    sum: f64,
    sum_squares: f64,
    bucket_limits: &[f64],
    bucket_counts: &[f64],
) -> Summary {
    let hist = HistogramProto {
        min,
        max,
        num,
        sum,
        sum_squares,
        bucket_limit: bucket_limits.to_vec(),
        bucket: bucket_counts.to_vec(),
    };

    let value = SummaryValue {
        tag: name.to_string(),
        value: Some(value::Value::Histo(hist)),
        ..SummaryValue::default()
    };

    Summary { value: vec![value] }
}

/// dim is in CHW
pub fn image(tag: &str, data: &[u8], dim: &[usize]) -> Summary {
    if dim.len() != 3 {
        panic!("format:CHW");
    }
    if dim[0] != 3 {
        panic!("needs rgb");
    }
    if data.len() != dim[0] * dim[1] * dim[2] {
        panic!("length of data should matches with dim.");
    }

    let mut img = RgbImage::new(dim[1] as u32, dim[2] as u32);
    img.clone_from_slice(data);
    let dimg = DynamicImage::ImageRgb8(img);
    let output_buf = Vec::<u8>::new();
    let mut c = Cursor::new(output_buf);
    dimg.write_to(&mut c, ImageFormat::Png).expect("");

    let output_image = SummaryImage {
        height: dim[1] as i32,
        width: dim[2] as i32,
        colorspace: 3,
        encoded_image_string: c.into_inner(),
    };
    let value = SummaryValue {
        tag: tag.to_string(),
        value: Some(value::Value::Image(output_image)),
        ..SummaryValue::default()
    };

    Summary { value: vec![value] }
}
