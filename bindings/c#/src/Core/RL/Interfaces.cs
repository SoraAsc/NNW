namespace NNW.Core.RL
{
    public interface IEnvironment
    {
        bool Step(uint action);
        uint Reset();
        uint GetState();
        float GetReward();
        uint GetNumStates();
        uint GetNumActions();
        bool IsDone();
    }
}