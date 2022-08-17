from src.CNN import TrainingLoop, ConvNet

trainer = TrainingLoop(ConvNet)
trainer.getDataset([1,2,3])
trainer.getModel()
trainer.predictionAndTest()
