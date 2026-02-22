using System;
using System.Runtime.InteropServices;
using NNW.Core.RL;

namespace NNW.Interop
{
    internal static class RLInterop
    {
        private const string LIB_NAME = "nn";

        static RLInterop()
        {
            NativeNNResolver.EnsureLoaded();
        }
        
        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr rl_create_qtable(UIntPtr states_num, UIntPtr actions_num);
        
        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rl_free_qtable(IntPtr qtable);
        
        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern float rl_get_qtable(IntPtr qtable, UIntPtr state, UIntPtr action);
        
        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rl_set_qtable(IntPtr qtable, UIntPtr state, UIntPtr action, float value);
        
        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr rl_get_qtable_states(IntPtr qtable);
        
        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr rl_get_qtable_actions(IntPtr qtable);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool rl_qtable_save(IntPtr qtable, [MarshalAs(UnmanagedType.LPStr)] string path);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr rl_qtable_load([MarshalAs(UnmanagedType.LPStr)] string path);


        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr rl_create_agent(UIntPtr states_num, UIntPtr actions_num, 
                                                     float learning_rate, float discount_factor);
        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rl_free_agent(IntPtr agent);
        
        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr rl_choose_agent_action(IntPtr agent, UIntPtr state);
        
        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rl_update_agent(IntPtr agent, UIntPtr state, UIntPtr action, 
                                                   float reward, UIntPtr next_state, int done);
        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rl_set_agent_policy(IntPtr agent, PolicyType policy_type, float epsilon);
        
        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern float rl_get_agent_learning_rate(IntPtr agent);
        
        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rl_set_agent_learning_rate(IntPtr agent, float lr);
        
        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern float rl_get_agent_discount_factor(IntPtr agent);
        
        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rl_set_agent_discount_factor(IntPtr agent, float gamma);
        
        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr rl_get_agent_qtable(IntPtr agent);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rl_set_agent_reward_clip(IntPtr agent, int enabled, float min_val, float max_val);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rl_set_agent_reward_normalization(IntPtr agent, int enabled, float scale);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern double rl_get_agent_average_reward(IntPtr agent);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern double rl_get_agent_last_reward(IntPtr agent);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr rl_get_agent_episode_count(IntPtr agent);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr rl_get_agent_last_episode_length(IntPtr agent);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern double rl_get_agent_average_episode_length(IntPtr agent);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rl_notify_agent_episode_end(IntPtr agent);
        
        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool rl_set_agent_epsilon_decay(IntPtr agent, double start, double min, double rate, int type, int per_step);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rl_update_agent_epsilon_step(IntPtr agent);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rl_update_agent_epsilon_episode(IntPtr agent);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern double rl_get_agent_epsilon(IntPtr agent);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rl_reset_agent_epsilon(IntPtr agent);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void rl_set_agent_training(IntPtr agent, int training);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int rl_get_agent_training(IntPtr agent);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool rl_save_agent(IntPtr agent, [MarshalAs(UnmanagedType.LPStr)] string path);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr rl_load_agent([MarshalAs(UnmanagedType.LPStr)] string path);
    }
}
