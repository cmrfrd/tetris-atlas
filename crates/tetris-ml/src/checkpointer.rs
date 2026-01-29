use anyhow::Result;
use candle_nn::VarMap;
use std::{fs, path::Path, path::PathBuf};
use tracing::info;

/// Manages periodic saving of model parameters to disk during training.
///
/// Checkpoints are saved as `{prefix}{run_name}_iter_{iteration:08}{.suffix}.safetensors`.
/// Supports automatic cleanup of old checkpoints based on retention policy.
pub struct Checkpointer {
    /// Save checkpoint every N iterations
    save_every: usize,
    /// Directory where checkpoints will be saved
    checkpoint_dir: PathBuf,
    /// Name of the training run (used in checkpoint filenames)
    run_name: String,
    /// Whether to keep all checkpoints or only the latest ones
    keep_all: bool,
    /// Maximum number of checkpoints to keep (if keep_all is false)
    max_checkpoints: usize,
    /// Optional default filename prefix
    default_prefix: Option<String>,
    /// Optional default filename suffix (before extension)
    default_suffix: Option<String>,
}

impl Checkpointer {
    /// Returns the checkpoint path for the given iteration using default prefix/suffix.
    pub fn checkpoint_path(&self, iteration: usize) -> PathBuf {
        self.checkpoint_path_with(
            iteration,
            self.default_prefix.as_deref(),
            self.default_suffix.as_deref(),
        )
    }
    /// Creates a new checkpointer with default settings.
    ///
    /// Defaults: keeps 3 most recent checkpoints, no prefix/suffix.
    ///
    /// # Errors
    /// Returns an error if the checkpoint directory cannot be created.
    pub fn new(save_every: usize, checkpoint_dir: PathBuf, run_name: String) -> Result<Self> {
        // Create checkpoint directory if it doesn't exist
        fs::create_dir_all(&checkpoint_dir)?;

        Ok(Self {
            save_every,
            checkpoint_dir,
            run_name,
            keep_all: false,
            max_checkpoints: 3,
            default_prefix: None,
            default_suffix: None,
        })
    }

    /// Creates a new checkpointer with custom retention and naming options.
    ///
    /// # Errors
    /// Returns an error if the checkpoint directory cannot be created.
    pub fn new_with_options(
        save_every: usize,
        checkpoint_dir: PathBuf,
        run_name: String,
        keep_all: bool,
        max_checkpoints: usize,
        default_prefix: Option<String>,
        default_suffix: Option<String>,
    ) -> Result<Self> {
        fs::create_dir_all(&checkpoint_dir)?;

        Ok(Self {
            save_every,
            checkpoint_dir,
            run_name,
            keep_all,
            max_checkpoints,
            default_prefix,
            default_suffix,
        })
    }

    /// Saves a checkpoint if the iteration is a multiple of `save_every`.
    ///
    /// Automatically cleans up old checkpoints based on retention policy.
    ///
    /// # Returns
    /// `true` if a checkpoint was saved, `false` otherwise.
    ///
    /// # Errors
    /// Returns an error if the checkpoint cannot be written or cleanup fails.
    pub fn checkpoint_item<T: Checkpointable + ?Sized>(
        &self,
        iteration: usize,
        item: &T,
        prefix: Option<&str>,
        suffix: Option<&str>,
    ) -> Result<bool> {
        if iteration > 0 && iteration.is_multiple_of(self.save_every) {
            let _ = self.save_item(iteration, item, prefix, suffix)?;
            if !self.keep_all {
                self.cleanup_old_checkpoints(iteration)?;
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Unconditionally saves a checkpoint regardless of save interval.
    ///
    /// Useful for saving final or best models. Does not trigger cleanup.
    ///
    /// # Errors
    /// Returns an error if the checkpoint cannot be written.
    pub fn force_checkpoint_item<T: Checkpointable + ?Sized>(
        &self,
        iteration: usize,
        item: &T,
        prefix: Option<&str>,
        suffix: Option<&str>,
    ) -> Result<()> {
        let _ = self.save_item(iteration, item, prefix, suffix)?;
        Ok(())
    }

    /// Saves a checkpoint and returns the path where it was saved.
    ///
    /// # Errors
    /// Returns an error if the checkpoint file cannot be written.
    pub fn save_item<T: Checkpointable + ?Sized>(
        &self,
        iteration: usize,
        item: &T,
        prefix: Option<&str>,
        suffix: Option<&str>,
    ) -> Result<PathBuf> {
        let path = self.checkpoint_path_with(iteration, prefix, suffix);
        item.save_to_path(&path)?;
        info!("Saved checkpoint at iteration {} to {:?}", iteration, path);
        Ok(path)
    }

    /// Loads the most recent checkpoint matching the optional suffix filter.
    ///
    /// # Returns
    /// `Some((path, iteration))` if a checkpoint was found and loaded, `None` otherwise.
    ///
    /// # Errors
    /// Returns an error if the checkpoint file cannot be read or is corrupted.
    pub fn load_item_latest<T: Checkpointable + ?Sized>(
        &self,
        item: &mut T,
        suffix: Option<&str>,
    ) -> Result<Option<(PathBuf, usize)>> {
        if let Some((path, iter)) = self.latest_checkpoint_filtered(suffix)? {
            item.load_from_path(&path)?;
            info!("Loaded checkpoint from iteration {} at {:?}", iter, path);
            Ok(Some((path, iter)))
        } else {
            Ok(None)
        }
    }

    /// Returns the checkpoint path for the given iteration with explicit prefix/suffix.
    pub fn checkpoint_path_with(
        &self,
        iteration: usize,
        prefix: Option<&str>,
        suffix: Option<&str>,
    ) -> PathBuf {
        let mut name = String::new();
        if let Some(p) = prefix {
            name.push_str(p);
        }
        name.push_str(&self.run_name);
        name.push_str("_iter_");
        name.push_str(&format!("{:08}", iteration));
        if let Some(s) = suffix
            && !s.is_empty()
        {
            name.push('.');
            name.push_str(s);
        }
        name.push_str(".safetensors");
        self.checkpoint_dir.join(name)
    }

    /// Finds the most recent checkpoint file and its iteration number.
    ///
    /// Excludes files containing ".optimizer." in the name.
    ///
    /// # Returns
    /// `Some((path, iteration))` if at least one checkpoint exists, `None` otherwise.
    pub fn latest_checkpoint(&self) -> Result<Option<(PathBuf, usize)>> {
        let mut latest_iteration = 0;
        let mut latest_path = None;

        if !self.checkpoint_dir.exists() {
            return Ok(None);
        }

        for entry in fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(filename) = path.file_name().and_then(|n| n.to_str())
                && filename.starts_with(&self.run_name)
                && filename.ends_with(".safetensors")
                && !filename.contains(".optimizer.")
            {
                // Expect formats:
                //   <run>_iter_XXXXXXXX.safetensors
                //   <run>_iter_XXXXXXXX.<suffix>.safetensors
                if let Some(after_prefix) =
                    filename.strip_prefix(&format!("{}_iter_", self.run_name))
                    && let Some(before_ext) = after_prefix.strip_suffix(".safetensors")
                {
                    let digits: String = before_ext
                        .chars()
                        .take_while(|c| c.is_ascii_digit())
                        .collect();
                    if !digits.is_empty()
                        && let Ok(iteration) = digits.parse::<usize>()
                        && iteration > latest_iteration
                    {
                        latest_iteration = iteration;
                        latest_path = Some(path.clone());
                    }
                }
            }
        }

        Ok(latest_path.map(|path| (path, latest_iteration)))
    }

    /// Finds the most recent checkpoint matching the optional suffix filter.
    ///
    /// # Returns
    /// `Some((path, iteration))` if a matching checkpoint exists, `None` otherwise.
    pub fn latest_checkpoint_filtered(
        &self,
        suffix: Option<&str>,
    ) -> Result<Option<(PathBuf, usize)>> {
        let mut latest_iteration = 0;
        let mut latest_path = None;

        if !self.checkpoint_dir.exists() {
            return Ok(None);
        }

        for entry in fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                if !filename.starts_with(&self.run_name) || !filename.ends_with(".safetensors") {
                    continue;
                }
                // Enforce suffix filter if provided.
                if let Some(sfx) = suffix {
                    let expect = format!(".{}.", sfx);
                    if !filename.contains(&expect) {
                        continue;
                    }
                }
                if let Some(after_prefix) =
                    filename.strip_prefix(&format!("{}_iter_", self.run_name))
                    && let Some(before_ext) = after_prefix.strip_suffix(".safetensors")
                {
                    let digits: String = before_ext
                        .chars()
                        .take_while(|c| c.is_ascii_digit())
                        .collect();
                    if !digits.is_empty()
                        && let Ok(iteration) = digits.parse::<usize>()
                        && iteration > latest_iteration
                    {
                        latest_iteration = iteration;
                        latest_path = Some(path.clone());
                    }
                }
            }
        }

        Ok(latest_path.map(|path| (path, latest_iteration)))
    }

    /// Loads checkpoint data from a specific file into a VarMap.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or has an invalid format.
    pub fn load_checkpoint(&self, checkpoint_path: &PathBuf, varmap: &mut VarMap) -> Result<()> {
        varmap.load(checkpoint_path)?;
        Ok(())
    }

    /// Loads the latest checkpoint into a VarMap.
    ///
    /// # Returns
    /// `Some(iteration)` if a checkpoint was found and loaded, `None` otherwise.
    ///
    /// # Errors
    /// Returns an error if the checkpoint exists but cannot be loaded.
    pub fn load_latest_checkpoint(&self, varmap: &mut VarMap) -> Result<Option<usize>> {
        if let Some((path, iteration)) = self.latest_checkpoint()? {
            self.load_checkpoint(&path, varmap)?;
            info!("Loaded checkpoint from iteration {}", iteration);
            Ok(Some(iteration))
        } else {
            info!("No checkpoint found for run '{}'", self.run_name);
            Ok(None)
        }
    }

    /// Removes old checkpoints, keeping only the most recent `max_checkpoints`.
    ///
    /// Called automatically by `checkpoint_item` when `keep_all` is false.
    fn cleanup_old_checkpoints(&self, _current_iteration: usize) -> Result<()> {
        let mut checkpoints = Vec::new();

        for entry in fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(filename) = path.file_name().and_then(|n| n.to_str())
                && filename.starts_with(&self.run_name)
                && filename.ends_with(".safetensors")
                && let Some(after_prefix) =
                    filename.strip_prefix(&format!("{}_iter_", self.run_name))
                && let Some(before_ext) = after_prefix.strip_suffix(".safetensors")
            {
                let digits: String = before_ext
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                if !digits.is_empty()
                    && let Ok(iteration) = digits.parse::<usize>()
                {
                    checkpoints.push((iteration, path.clone()));
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

/// Trait for types that can be saved to and loaded from checkpoint files.
///
/// Implement this trait to enable custom types to work with [`Checkpointer`].
pub trait Checkpointable {
    /// Saves the object's state to the specified path.
    fn save_to_path(&self, path: &Path) -> candle_core::Result<()>;

    /// Loads the object's state from the specified path.
    fn load_from_path(&mut self, path: &Path) -> candle_core::Result<()>;
}

impl Checkpointable for VarMap {
    fn save_to_path(&self, path: &Path) -> candle_core::Result<()> {
        self.save(path)
    }
    fn load_from_path(&mut self, path: &Path) -> candle_core::Result<()> {
        self.load(path)
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
