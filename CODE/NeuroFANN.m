clc;clear;

addpath("./Function");

load('Sample.mat');
Parameter.NumIter = 100;
Parameter.NumFold = 5;
Parameter.MaxEpoch = 500;
Parameter.LearnRate = 0.001;
Parameter.RegCoeff = 0.005;

ScoreABT = cell(Parameter.NumIter,Parameter.NumFold);
ScoreMTA = cell(Parameter.NumIter,Parameter.NumFold);
ScoreWMH = cell(Parameter.NumIter,Parameter.NumFold);

for IdxIter = 1:Parameter.NumIter

    RandStream.setGlobalStream(RandStream('mt19937ar','Seed',IdxIter))
    MdlIter.IdxIter = IdxIter; MdlIter.Dataset = Dataset; MdlIter.Parameter = Parameter;

    for IdxFold = 1:MdlIter.Parameter.NumFold
        MdlFold = ModelInit(MdlIter,IdxFold);
        MdlFold = ParamInit(MdlFold);
        MdlFold.AdamParam = AdamInit(MdlFold.NumParam,MdlFold.LearnRate);
        MdlFold = ParamTrain(MdlFold);
        ScoreABT{IdxIter,IdxFold} = MdlFold.PabtTest;
        ScoreMTA{IdxIter,IdxFold} = MdlFold.PmtaTest;
        ScoreWMH{IdxIter,IdxFold} = MdlFold.PwmhTest;
    end

end