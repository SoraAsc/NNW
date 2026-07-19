namespace NNW.Core.NN
{
    public enum Optimizer : int
    {
        SGD = 0,
        ADAMW = 1
    }

    public enum Loss : int
    {
        MSE = 0,
    }

    public enum Activation : int
    {
        LINEAR = 0,
        RELU = 1,
        SIGMOID = 2,
        TANH = 3
    }
}
