import fire
from src.dosImp import DOSProcedure

def main():
    selectedClases = range(10)
    kj = {i:2 for i in range(10)}
    rj = {i:2 for i in range(10)}
    dos = DOSProcedure(selectedClases, kj, rj, maxThresh=100)
    dos.mainLoop()
    return dos.labelOverSampledTupsMap

if __name__=="__main__":
    lOSTMap = main()
    # fire.Fire(main) 
