import fire
from src.mtl import TrainingLoopDOS
from src.dosImp import ConvNet, applyEmbedder, applyClassifier

def main(epochs, batch_size, learning_rate=0.0001):
    epochs=int(epochs)
    batch_size=int(batch_size)
    learning_rate=float(learning_rate)

    selectedClases = range(10)
    kj = {i:2 for i in range(10)}
    rj = {i:2 for i in range(10)}

    trainer = TrainingLoopDOS(selectedClases, kj, rj, maxThresh=100)
    trainer.trainingLoop(applyEmbedder, applyClassifier)
    # trainer.getModel(learning_rate=learning_rate, epochs=epochs, batch_size=batch_size)
    # trainer.predictionAndTest()

if __name__=="__main__":
   fire.Fire(main) 
