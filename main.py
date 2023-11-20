import fire
from src.mtl import TrainingLoopDOS
from src.dosImp import ConvNet, applyEmbedder, applyClassifier
from src.newCNN import TrainingLoop

def main(epochs, batch_size, maxThresh, learning_rate=0.0001):
    epochs=int(epochs)
    batch_size=int(batch_size)
    learning_rate=float(learning_rate)

    # selectedClases = range(10)
    # selectedClases = [4,6,8]
    selectedClases = []
    kj = {i:0 for i in range(10)}
    rj = {i:0 for i in range(10)}

    initialTrainer = TrainingLoop(selectedClases, kj, rj, maxThresh)
    initialTrainer.getModel(applyEmbedder, applyClassifier,
                            learning_rate=learning_rate,
                            epochs=epochs, batch_size=batch_size)
    initialTrainer.predictionAndTest()
    # trainer = TrainingLoopDOS(selectedClases, kj, rj, maxThresh=maxThresh)
    # trainer.trainingLoop(applyEmbedder, applyClassifier, epochs=epochs, 
    #                      batch_size=batch_size, learning_rate=learning_rate)
    # trainer.predictionAndTest()
if __name__=="__main__":
   fire.Fire(main) 
