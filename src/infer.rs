use burn::{
    data::dataloader::batcher::Batcher,
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{Tensor, backend::Backend},
};

use crate::{
    model::TetrisGameTransformer,
    tetris::{TetrisBoardRaw, TetrisPiecePlacement},
};

pub fn infer<B: Backend>(
    artifact_dir: &str,
    device: B::Device,
    board: TetrisBoardRaw,
    placement: TetrisPiecePlacement,
) {
    // let record = CompactRecorder::new()
    //     .load(format!("{artifact_dir}/model").into(), &device)
    //     .expect("Trained model should exist");

    let input_board = Tensor::<B, 1>::from_floats(board.to_binary_slice(), &device);
    let input_placement = Tensor::<B, 1>::from_floats([placement.index()], &device);

    // let model: TetrisGameTransformer<B> = TetrisGameTransformer::new(NUM_CLASSES.into(), &device).load_record(record);

    // let mut label = 0;
    // if let Annotation::Label(category) = item.annotation {
    //     label = category;
    // };
    // let batcher = ClassificationBatcher::new(device);
    // let batch = batcher.batch(vec![item]);
    // let output = model.forward(batch.images);
    // let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();
    // println!("Predicted {predicted} Expected {label:?}");
}
