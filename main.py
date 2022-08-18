import fire
from src.CNN import TrainingLoop
from src.dosImp import ConvNet

def main(epochs, batch_size, learning_rate=0.0001):
    epochs=int(epochs)
    batch_size=int(batch_size)
    learning_rate=float(learning_rate)

    trainer = TrainingLoop(ConvNet, selectedClases=[1,2,3])
    trainer.getModel(learning_rate=learning_rate, epochs=epochs, batch_size=batch_size)
    trainer.predictionAndTest()

if __name__=="__main__":
   fire.Fire(main) 
