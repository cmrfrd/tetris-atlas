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
