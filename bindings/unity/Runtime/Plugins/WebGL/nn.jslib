mergeInto(LibraryManager.library, {
  // NN Model API
  nn_create_model: function(input_dim) {
    return Module.ccall('nn_create_model', 'number', ['number'], [input_dim]);
  },
  nn_free_model: function(model) {
    Module.ccall('nn_free_model', 'void', ['number'], [model]);
  },
  nn_add_dense: function(model, units, act) {
    Module.ccall('nn_add_dense', 'void', ['number', 'number', 'number'], [model, units, act]);
  },
  nn_get_input_dim: function(model) {
    return Module.ccall('nn_get_input_dim', 'number', ['number'], [model]);
  },
  nn_get_output_dim: function(model) {
    return Module.ccall('nn_get_output_dim', 'number', ['number'], [model]);
  },
  nn_create_trainer: function(model, optimizer, loss, cfg) {
    return Module.ccall('nn_create_trainer', 'number', ['number', 'number', 'number', 'number'], [model, optimizer, loss, cfg]);
  },
  nn_free_trainer: function(trainer) {
    Module.ccall('nn_free_trainer', 'void', ['number'], [trainer]);
  },
  nn_train_fit: function(trainer, x, n_samples, x_dim, y, y_dim) {
    Module.ccall('nn_train_fit', 'void', ['number', 'array', 'number', 'number', 'array', 'number'], [trainer, x, n_samples, x_dim, y, y_dim]);
  },
  nn_predict: function(model, x, n_samples, x_dim, out, y_dim) {
    Module.ccall('nn_predict', 'void', ['number', 'array', 'number', 'number', 'array', 'number'], [model, x, n_samples, x_dim, out, y_dim]);
  },

  // RL Q-Table API
  rl_create_qtable: function(states_num, actions_num) {
    return Module.ccall('rl_create_qtable', 'number', ['number', 'number'], [states_num, actions_num]);
  },
  rl_free_qtable: function(qtable) {
    Module.ccall('rl_free_qtable', 'void', ['number'], [qtable]);
  },
  rl_get_qtable: function(qtable, state, action) {
    return Module.ccall('rl_get_qtable', 'number', ['number', 'number', 'number'], [qtable, state, action]);
  },
  rl_set_qtable: function(qtable, state, action, value) {
    Module.ccall('rl_set_qtable', 'void', ['number', 'number', 'number', 'number'], [qtable, state, action, value]);
  },
  rl_get_qtable_states: function(qtable) {
    return Module.ccall('rl_get_qtable_states', 'number', ['number'], [qtable]);
  },
  rl_get_qtable_actions: function(qtable) {
    return Module.ccall('rl_get_qtable_actions', 'number', ['number'], [qtable]);
  },
  rl_qtable_save: function(qtable, path) {
    return Module.ccall('rl_qtable_save', 'number', ['number', 'string'], [qtable, path]);
  },
  rl_qtable_load: function(path) {
    return Module.ccall('rl_qtable_load', 'number', ['string'], [path]);
  },

  // RL Agent API
  rl_create_agent: function(states_num, actions_num, learning_rate, discount_factor) {
    return Module.ccall('rl_create_agent', 'number', ['number', 'number', 'number', 'number'], [states_num, actions_num, learning_rate, discount_factor]);
  },
  rl_free_agent: function(agent) {
    Module.ccall('rl_free_agent', 'void', ['number'], [agent]);
  },
  rl_choose_agent_action: function(agent, state) {
    return Module.ccall('rl_choose_agent_action', 'number', ['number', 'number'], [agent, state]);
  },
  rl_update_agent: function(agent, state, action, reward, next_state, done) {
    Module.ccall('rl_update_agent', 'void', ['number', 'number', 'number', 'number', 'number', 'number'], [agent, state, action, reward, next_state, done]);
  },
  rl_set_agent_policy: function(agent, policy_type, epsilon) {
    Module.ccall('rl_set_agent_policy', 'void', ['number', 'number', 'number'], [agent, policy_type, epsilon]);
  },
  rl_get_agent_learning_rate: function(agent) {
    return Module.ccall('rl_get_agent_learning_rate', 'number', ['number'], [agent]);
  },
  rl_set_agent_learning_rate: function(agent, lr) {
    Module.ccall('rl_set_agent_learning_rate', 'void', ['number', 'number'], [agent, lr]);
  },
  rl_get_agent_discount_factor: function(agent) {
    return Module.ccall('rl_get_agent_discount_factor', 'number', ['number'], [agent]);
  },
  rl_set_agent_discount_factor: function(agent, gamma) {
    Module.ccall('rl_set_agent_discount_factor', 'void', ['number', 'number'], [agent, gamma]);
  },
  rl_get_agent_qtable: function(agent) {
    return Module.ccall('rl_get_agent_qtable', 'number', ['number'], [agent]);
  },
  rl_set_agent_reward_clip: function(agent, enabled, min_val, max_val) {
    Module.ccall('rl_set_agent_reward_clip', 'void', ['number', 'number', 'number', 'number'], [agent, enabled, min_val, max_val]);
  },
  rl_set_agent_reward_normalization: function(agent, enabled, scale) {
    Module.ccall('rl_set_agent_reward_normalization', 'void', ['number', 'number', 'number'], [agent, enabled, scale]);
  },
  rl_get_agent_average_reward: function(agent) {
    return Module.ccall('rl_get_agent_average_reward', 'number', ['number'], [agent]);
  },
  rl_get_agent_last_reward: function(agent) {
    return Module.ccall('rl_get_agent_last_reward', 'number', ['number'], [agent]);
  },
  rl_get_agent_episode_count: function(agent) {
    return Module.ccall('rl_get_agent_episode_count', 'number', ['number'], [agent]);
  },
  rl_get_agent_last_episode_length: function(agent) {
    return Module.ccall('rl_get_agent_last_episode_length', 'number', ['number'], [agent]);
  },
  rl_get_agent_average_episode_length: function(agent) {
    return Module.ccall('rl_get_agent_average_episode_length', 'number', ['number'], [agent]);
  },
  rl_notify_agent_episode_end: function(agent) {
    Module.ccall('rl_notify_agent_episode_end', 'void', ['number'], [agent]);
  },
  rl_set_agent_epsilon_decay: function(agent, start, min, rate, type, per_step) {
    return Module.ccall('rl_set_agent_epsilon_decay', 'number', ['number', 'number', 'number', 'number', 'number', 'number'], [agent, start, min, rate, type, per_step]);
  },
  rl_update_agent_epsilon_step: function(agent) {
    Module.ccall('rl_update_agent_epsilon_step', 'void', ['number'], [agent]);
  },
  rl_update_agent_epsilon_episode: function(agent) {
    Module.ccall('rl_update_agent_epsilon_episode', 'void', ['number'], [agent]);
  },
  rl_get_agent_epsilon: function(agent) {
    return Module.ccall('rl_get_agent_epsilon', 'number', ['number'], [agent]);
  },
  rl_reset_agent_epsilon: function(agent) {
    Module.ccall('rl_reset_agent_epsilon', 'void', ['number'], [agent]);
  },
  rl_set_agent_training: function(agent, training) {
    Module.ccall('rl_set_agent_training', 'void', ['number', 'number'], [agent, training]);
  },
  rl_get_agent_training: function(agent) {
    return Module.ccall('rl_get_agent_training', 'number', ['number'], [agent]);
  },
  rl_save_agent: function(agent, path) {
    return Module.ccall('rl_save_agent', 'number', ['number', 'string'], [agent, path]);
  },
  rl_load_agent: function(path) {
    return Module.ccall('rl_load_agent', 'number', ['string'], [path]);
  }
});
