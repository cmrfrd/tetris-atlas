use anyhow::Result;
use candle_nn::VarMap;
use std::{fs, path::PathBuf};

/// A simple checkpointer for saving model state during training.
///
/// This struct manages periodic saving of model parameters to disk,
/// allowing training to be resumed from the latest checkpoint.
pub struct Checkpointer {
    /// Save checkpoint every N iterations
    save_every: usize,
    /// Directory where checkpoints will be saved
    checkpoint_dir: PathBuf,
    /// Name of the training run (used in checkpoint filenames)
    run_name: String,
    /// Whether to keep all checkpoints or only the latest one
    keep_all: bool,
    /// Maximum number of checkpoints to keep (if keep_all is false)
    max_checkpoints: usize,
}

impl Checkpointer {
    /// Create a new checkpointer.
    pub fn new(save_every: usize, checkpoint_dir: PathBuf, run_name: String) -> Result<Self> {
        // Create checkpoint directory if it doesn't exist
        fs::create_dir_all(&checkpoint_dir)?;

        Ok(Self {
            save_every,
            checkpoint_dir,
            run_name,
            keep_all: false,
            max_checkpoints: 3,
        })
    }

    /// Create a new checkpointer with additional options.
    pub fn new_with_options(
        save_every: usize,
        checkpoint_dir: PathBuf,
        run_name: String,
        keep_all: bool,
        max_checkpoints: usize,
    ) -> Result<Self> {
        fs::create_dir_all(&checkpoint_dir)?;

        Ok(Self {
            save_every,
            checkpoint_dir,
            run_name,
            keep_all,
            max_checkpoints,
        })
    }

    /// Save a checkpoint if the current iteration matches the save interval.
    pub fn checkpoint(&self, iteration: usize, varmap: &VarMap) -> Result<bool> {
        if iteration > 0 && iteration % self.save_every == 0 {
            self.save_checkpoint(iteration, varmap)?;

            if !self.keep_all {
                self.cleanup_old_checkpoints(iteration)?;
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Force save a checkpoint regardless of the save interval.
    pub fn force_checkpoint(&self, iteration: usize, varmap: &VarMap) -> Result<()> {
        self.save_checkpoint(iteration, varmap)
    }

    /// Get the path for a checkpoint file at the given iteration.
    pub fn checkpoint_path(&self, iteration: usize) -> PathBuf {
        let filename = format!("{}_iter_{:08}.safetensors", self.run_name, iteration);
        self.checkpoint_dir.join(filename)
    }

    /// Get the latest checkpoint path and iteration number.
    pub fn latest_checkpoint(&self) -> Result<Option<(PathBuf, usize)>> {
        let mut latest_iteration = 0;
        let mut latest_path = None;

        if !self.checkpoint_dir.exists() {
            return Ok(None);
        }

        for entry in fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                if filename.starts_with(&self.run_name) && filename.ends_with(".safetensors") {
                    // Parse iteration number from filename
                    if let Some(iter_str) = filename
                        .strip_prefix(&format!("{}_iter_", self.run_name))
                        .and_then(|s| s.strip_suffix(".safetensors"))
                    {
                        if let Ok(iteration) = iter_str.parse::<usize>() {
                            if iteration > latest_iteration {
                                latest_iteration = iteration;
                                latest_path = Some(path);
                            }
                        }
                    }
                }
            }
        }

        Ok(latest_path.map(|path| (path, latest_iteration)))
    }

    /// Load a checkpoint from the given path into the VarMap.
    pub fn load_checkpoint(&self, checkpoint_path: &PathBuf, varmap: &mut VarMap) -> Result<()> {
        varmap.load(checkpoint_path)?;
        Ok(())
    }

    /// Load the latest checkpoint if it exists.
    pub fn load_latest_checkpoint(&self, varmap: &mut VarMap) -> Result<Option<usize>> {
        if let Some((path, iteration)) = self.latest_checkpoint()? {
            self.load_checkpoint(&path, varmap)?;
            println!("Loaded checkpoint from iteration {}", iteration);
            Ok(Some(iteration))
        } else {
            println!("No checkpoint found for run '{}'", self.run_name);
            Ok(None)
        }
    }

    /// Internal method to save a checkpoint.
    fn save_checkpoint(&self, iteration: usize, varmap: &VarMap) -> Result<()> {
        let checkpoint_path = self.checkpoint_path(iteration);
        varmap.save(&checkpoint_path)?;
        println!(
            "Saved checkpoint at iteration {} to {:?}",
            iteration, checkpoint_path
        );
        Ok(())
    }

    /// Clean up old checkpoints, keeping only the most recent ones.
    fn cleanup_old_checkpoints(&self, current_iteration: usize) -> Result<()> {
        let mut checkpoints = Vec::new();

        for entry in fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                if filename.starts_with(&self.run_name) && filename.ends_with(".safetensors") {
                    // Parse iteration number from filename
                    if let Some(iter_str) = filename
                        .strip_prefix(&format!("{}_iter_", self.run_name))
                        .and_then(|s| s.strip_suffix(".safetensors"))
                    {
                        if let Ok(iteration) = iter_str.parse::<usize>() {
                            checkpoints.push((iteration, path));
                        }
                    }
                }
            }
        }

        // Sort by iteration number (newest first)
        checkpoints.sort_by(|a, b| b.0.cmp(&a.0));

        // Remove old checkpoints beyond max_checkpoints
        if checkpoints.len() > self.max_checkpoints {
            for (_, path) in checkpoints.iter().skip(self.max_checkpoints) {
                if let Err(e) = fs::remove_file(path) {
                    eprintln!("Warning: Failed to remove old checkpoint {:?}: {}", path, e);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpointer_creation() {
        use std::env;
        let temp_dir = env::temp_dir().join("test_checkpointer");
        let checkpointer = Checkpointer::new(10, temp_dir.clone(), "test_run".to_string()).unwrap();

        assert_eq!(checkpointer.save_every, 10);
        assert_eq!(checkpointer.run_name, "test_run");
        assert!(temp_dir.exists());
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_checkpoint_path() {
        use std::env;
        let temp_dir = env::temp_dir().join("test_checkpointer2");
        let checkpointer = Checkpointer::new(10, temp_dir.clone(), "test_run".to_string()).unwrap();

        let path = checkpointer.checkpoint_path(1000);
        let expected = temp_dir.join("test_run_iter_00001000.safetensors");
        assert_eq!(path, expected);
        let _ = std::fs::remove_dir_all(temp_dir);
    }
}
