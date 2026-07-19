namespace NNW.Core.RL
{
    public interface IEnvironment<TAgent>
    {
        void Step(TAgent agent, uint action);
        uint GetState(TAgent agent);
        float GetReward(TAgent agent);
        bool IsEnvDone();
        void ResetEnv();
        uint GetNumStates();
        uint GetNumActions();
    }
}