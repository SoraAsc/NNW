namespace NNW.Core.RL
{
    public enum PolicyType: int
    {
        Greedy = 0,
        EpsilonGreedy = 1
    }

    public enum EpsilonDecayType: int
    {
        Linear = 0,
        Exponential = 1
    }
}