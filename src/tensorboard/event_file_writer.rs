use gethostname::gethostname;
use prost::Message;
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::id;
use std::sync::mpsc::{Sender, channel};
use std::thread::{JoinHandle, spawn};
use std::time::SystemTime;

use crate::tensorboard::record_writer::RecordWriter;
use crate::tensorboard_generated::tensorboard::Event;
use crate::tensorboard_generated::tensorboard::event::What;

enum EventSignal {
    Data(Vec<u8>),
    Flush,
    Stop,
}

pub struct EventFileWriter {
    logdir: PathBuf,
    writer: Sender<EventSignal>,
    child: Option<JoinHandle<()>>,
}
impl EventFileWriter {
    //pub fn new<P: AsRef<Path>>(logdir: P) -> EventFileWriter {
    pub fn new<P: AsRef<Path>>(logdir: P) -> EventFileWriter {
        let logdir = logdir.as_ref().to_path_buf();

        fs::create_dir_all(&logdir).expect("");

        let mut time = 0;
        let mut time_full = 0.0;
        if let Ok(n) = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
            time = n.as_secs();
            time_full = n.as_secs_f64();
        }
        let hostname = gethostname().into_string().expect("");
        let pid = id();

        let file_name = format!(
            "events.out.tfevents.{:010}.{}.{}.{}",
            time, hostname, pid, 0
        );
        //let file_writer = File::create(logdir.join(file_name)).expect("");
        //let writer = RecordWriter::new(file_writer);

        let logdir_move = logdir.clone();
        let (tx, rx) = channel();
        let child = spawn(move || {
            let file_writer = File::create(logdir_move.join(file_name)).expect("");
            let mut writer = RecordWriter::new(file_writer);

            loop {
                let result: EventSignal = rx.recv().unwrap();
                match result {
                    EventSignal::Data(d) => {
                        writer.write(&d).expect("write error");
                    }
                    EventSignal::Flush => {
                        writer.flush().expect("flush error");
                    }
                    EventSignal::Stop => {
                        break;
                    }
                }
            }
            writer.flush().expect("flush error");
        });

        let mut ret = EventFileWriter {
            logdir,
            writer: tx,
            child: Some(child),
        };

        let mut evn = Event::default();
        evn.wall_time = time_full;
        evn.what = Some(What::FileVersion("brain.Event:2".to_string()));

        ret.add_event(&evn);
        ret.flush();

        ret
    }
}

impl EventFileWriter {
    pub fn get_logdir(&self) -> PathBuf {
        self.logdir.to_path_buf()
    }

    pub fn add_event(&mut self, event: &Event) {
        let mut data: Vec<u8> = event.encode_to_vec();
        self.writer.send(EventSignal::Data(data)).expect("");
    }

    pub fn flush(&mut self) {
        self.writer.send(EventSignal::Flush).expect("");
    }
}

impl Drop for EventFileWriter {
    fn drop(&mut self) {
        self.writer.send(EventSignal::Stop).expect("");
        self.child.take().unwrap().join().expect("");
    }
}
