#![allow(clippy::too_many_arguments)]
use prost::Message;

use crate::tensorboard::event_file_writer::EventFileWriter;
use crate::tensorboard::summary::{histogram_raw, image, scalar};
use crate::tensorboard_generated::tensorboard::GraphDef;
use crate::tensorboard_generated::tensorboard::NodeDef;
use crate::tensorboard_generated::tensorboard::RunMetadata;
use crate::tensorboard_generated::tensorboard::Summary;
use crate::tensorboard_generated::tensorboard::VersionDef;
use crate::tensorboard_generated::tensorboard::event::What;
use crate::tensorboard_generated::tensorboard::{Event, TaggedRunMetadata};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

pub struct FileWriter {
    writer: EventFileWriter,
}

impl FileWriter {
    #[allow(dead_code)]
    pub(crate) fn add_global_summary(&mut self, summary: Summary) {
        let mut evn = Event::default();
        evn.what = Some(What::Summary(summary));
        self.writer.add_event(&evn);
    }
}

impl FileWriter {
    pub fn new<P: AsRef<Path>>(logdir: P) -> FileWriter {
        FileWriter {
            writer: EventFileWriter::new(logdir),
        }
    }

    pub fn get_logdir(&self) -> PathBuf {
        self.writer.get_logdir()
    }

    pub fn add_event(&mut self, event: &Event, step: usize) {
        let mut event = event.clone();

        let mut time_full = 0.0;
        if let Ok(n) = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
            time_full = n.as_secs_f64();
        }
        event.wall_time = time_full;

        event.step = step as i64;

        self.writer.add_event(&event)
    }

    pub fn add_summary(&mut self, summary: Summary, step: usize) {
        let mut evn = Event::default();
        evn.what = Some(What::Summary(summary));
        self.add_event(&evn, step)
    }

    pub fn add_graph(&mut self, graph: GraphDef, meta: RunMetadata) {
        let graph_vec = graph.encode_to_vec();
        let mut graph_evn = Event::default();
        graph_evn.what = Some(What::GraphDef(graph_vec));
        self.writer.add_event(&graph_evn);

        let meta_vec = meta.encode_to_vec();
        let mut tagged_meta = TaggedRunMetadata::default();
        tagged_meta.tag = "profiler".to_string();
        tagged_meta.run_metadata = meta_vec;
        let mut meta_evn = Event::default();
        meta_evn.what = Some(What::TaggedRunMetadata(tagged_meta));
        self.writer.add_event(&meta_evn);
    }

    pub fn flush(&mut self) {
        self.writer.flush()
    }
}

pub struct SummaryWriter {
    writer: FileWriter,
    all_writers: HashMap<PathBuf, FileWriter>,
}

impl SummaryWriter {
    pub fn new<P: AsRef<Path>>(logdir: P) -> SummaryWriter {
        SummaryWriter {
            writer: FileWriter::new(logdir),
            all_writers: HashMap::new(),
        }
    }

    pub fn add_scalar(&mut self, tag: &str, scalar_value: f32, step: usize) {
        self.writer.add_summary(scalar(tag, scalar_value), step);
    }

    pub fn add_scalars(&mut self, main_tag: &str, tag_scalar: &HashMap<String, f32>, step: usize) {
        let base_logdir = self.writer.get_logdir();
        for (tag, scalar_value) in tag_scalar.iter() {
            let fw_tag = base_logdir.join(main_tag).join(tag);
            if !self.all_writers.contains_key(&fw_tag) {
                let new_writer = FileWriter::new(fw_tag.clone());
                self.all_writers.insert(fw_tag.clone(), new_writer);
            }
            let fw = self.all_writers.get_mut(&fw_tag).expect("");
            fw.add_summary(scalar(main_tag, *scalar_value), step);
        }
    }

    /// Automatically determines optimal number of histogram buckets using statistical rules.
    ///
    /// Uses the Freedman-Diaconis rule which is robust to outliers:
    /// bin_width = 2 * IQR / n^(1/3), where IQR is the interquartile range.
    ///
    /// Falls back to Sturges' rule for small datasets: k = ceil(log2(n) + 1)
    fn auto_buckets(data: &[f64]) -> usize {
        let n = data.len();

        // For very small datasets, use simple rules
        if n < 10 {
            return n.min(5).max(1);
        }

        // Sturges' rule as baseline (works well for normal distributions)
        let sturges = ((n as f64).log2() + 1.0).ceil() as usize;

        // Try Freedman-Diaconis rule (more robust to outliers)
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1_idx = n / 4;
        let q3_idx = (3 * n) / 4;
        let q1 = sorted_data[q1_idx];
        let q3 = sorted_data[q3_idx];
        let iqr = q3 - q1;

        let min = sorted_data[0];
        let max = sorted_data[n - 1];
        let range = max - min;

        // If IQR is too small or zero, fall back to Sturges
        if iqr < 1e-10 || range < 1e-10 {
            return sturges.clamp(10, 50);
        }

        // Freedman-Diaconis: bin_width = 2 * IQR * n^(-1/3)
        let bin_width = 2.0 * iqr / (n as f64).powf(1.0 / 3.0);
        let fd_buckets = (range / bin_width).ceil() as usize;

        // Clamp to reasonable range (10-100) and prefer FD rule but consider Sturges too
        let buckets = fd_buckets.max(sturges).clamp(10, 100);

        buckets
    }

    pub fn add_histogram_raw(
        &mut self,
        tag: &str,
        min: f64,
        max: f64,
        num: f64,
        sum: f64,
        sum_squares: f64,
        bucket_limits: &[f64],
        bucket_counts: &[f64],
        step: usize,
    ) {
        if bucket_limits.len() != bucket_counts.len() {
            panic!("bucket_limits.len() != bucket_counts.len()");
        }

        self.writer.add_summary(
            histogram_raw(
                tag,
                min,
                max,
                num,
                sum,
                sum_squares,
                bucket_limits,
                bucket_counts,
            ),
            step,
        );
    }

    /// Adds a histogram from raw data, automatically calculating all necessary statistics.
    ///
    /// This is a convenience method that computes min, max, sum, sum_squares, and bucket
    /// distributions from the provided data.
    ///
    /// # Arguments
    /// * `tag` - The tag/name for this histogram
    /// * `data` - Slice of values to create the histogram from
    /// * `num_buckets` - Number of buckets to use. If None, automatically determines optimal count
    ///   using the Freedman-Diaconis rule (or Sturges' rule as fallback)
    /// * `step` - The global step value
    pub fn add_histogram(
        &mut self,
        tag: &str,
        data: &[f64],
        num_buckets: Option<usize>,
        step: usize,
    ) {
        if data.is_empty() {
            return;
        }

        let num_buckets = num_buckets.unwrap_or_else(|| Self::auto_buckets(data));

        // Calculate basic statistics
        let min = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = data.iter().sum();
        let sum_squares: f64 = data.iter().map(|x| x * x).sum();
        let num = data.len() as f64;

        // Handle edge case where all values are the same
        if (max - min).abs() < f64::EPSILON {
            let bucket_limits = vec![min + 1e-10];
            let bucket_counts = vec![num];
            self.add_histogram_raw(
                tag,
                min,
                max,
                num,
                sum,
                sum_squares,
                &bucket_limits,
                &bucket_counts,
                step,
            );
            return;
        }

        // Create bucket limits and counts
        let mut bucket_limits = Vec::with_capacity(num_buckets);
        let mut bucket_counts = vec![0.0; num_buckets];

        let bucket_width = (max - min) / num_buckets as f64;
        for i in 0..num_buckets {
            bucket_limits.push(min + (i as f64 + 1.0) * bucket_width);
        }

        // Count values in each bucket
        for &value in data {
            let bucket_idx = ((value - min) / bucket_width).floor() as usize;
            // Handle edge case where value == max
            let bucket_idx = bucket_idx.min(num_buckets - 1);
            bucket_counts[bucket_idx] += 1.0;
        }

        self.add_histogram_raw(
            tag,
            min,
            max,
            num,
            sum,
            sum_squares,
            &bucket_limits,
            &bucket_counts,
            step,
        );
    }

    pub fn add_image(&mut self, tag: &str, data: &[u8], dim: &[usize], step: usize) {
        self.writer.add_summary(image(tag, data, dim), step);
    }

    pub fn add_graph(&mut self, node_list: &[NodeDef]) {
        let mut graph = GraphDef::default();

        let nodes = node_list.to_vec();
        graph.node = nodes;

        let mut version = VersionDef::default();
        version.producer = 22;
        graph.versions = Some(version);

        let stats = RunMetadata::default();

        self.writer.add_graph(graph, stats);
    }

    pub fn flush(&mut self) {
        self.writer.flush();
    }
}
