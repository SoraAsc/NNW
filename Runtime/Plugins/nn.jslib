mergeInto(LibraryManager.library, {

  $NNState: { module: null, ready: false, loading: false },

  nn_is_ready__sig: 'i',
  nn_is_ready__deps: ['$NNState'],
  nn_is_ready: function () {
    return NNState.ready ? 1 : 0;
  },

  nn_init_module__sig: 'v',
  nn_init_module__deps: ['$NNState'],
  nn_init_module: function () {
    if (NNState.loading || NNState.ready) return;
    NNState.loading = true;
    var base = (typeof Module !== 'undefined' && Module.streamingAssetsUrl)
      ? Module.streamingAssetsUrl + '/NNW/'
      : 'StreamingAssets/NNW/';

    var script = document.createElement('script');
    script.src = base + 'nn.js';
    script.onload = function () {
      createNNModule({
        locateFile: function (f) { return base + f; }
      }).then(function (mod) {
        NNState.module = mod;
        NNState.ready  = true;
        console.log('[NNW] ready at ' + base);
      }).catch(function (e) {
        console.error('[NNW] createNNModule failed:', e);
      });
    };
    script.onerror = function () {
      console.error('[NNW] Failed to load: ' + script.src);
    };
    document.head.appendChild(script);
  },

  // NN
  nn_create_model__sig: 'ii',
  nn_create_model__deps: ['$NNState'],
  nn_create_model: function (input_dim) {
    return NNState.module._nn_create_model(input_dim);
  },

  nn_free_model__sig: 'vi',
  nn_free_model__deps: ['$NNState'],
  nn_free_model: function (model) {
    NNState.module._nn_free_model(model);
  },

  nn_add_dense__sig: 'viii',
  nn_add_dense__deps: ['$NNState'],
  nn_add_dense: function (model, units, act) {
    NNState.module._nn_add_dense(model, units, act);
  },

  nn_get_input_dim__sig: 'ii',
  nn_get_input_dim__deps: ['$NNState'],
  nn_get_input_dim: function (model) {
    return NNState.module._nn_get_input_dim(model);
  },

  nn_get_output_dim__sig: 'ii',
  nn_get_output_dim__deps: ['$NNState'],
  nn_get_output_dim: function (model) {
    return NNState.module._nn_get_output_dim(model);
  },

  nn_create_trainer__sig: 'iiiii',
  nn_create_trainer__deps: ['$NNState'],
  nn_create_trainer: function (model, optimizer, loss, cfg) {
    return NNState.module._nn_create_trainer(model, optimizer, loss, cfg);
  },

  nn_free_trainer__sig: 'vi',
  nn_free_trainer__deps: ['$NNState'],
  nn_free_trainer: function (trainer) {
    NNState.module._nn_free_trainer(trainer);
  },

  nn_train_fit__sig: 'viiiiiii',
  nn_train_fit__deps: ['$NNState'],
  nn_train_fit: function (trainer, x, n_samples, x_dim, y, y_dim) {
    NNState.module._nn_train_fit(trainer, x, n_samples, x_dim, y, y_dim);
  },

  nn_predict__sig: 'viiiiii',
  nn_predict__deps: ['$NNState'],
  nn_predict: function (model, x, n_samples, x_dim, outBuf, y_dim) {
    NNState.module._nn_predict(model, x, n_samples, x_dim, outBuf, y_dim);
  },

  // RL - QTable
  rl_create_qtable__sig: 'iii',
  rl_create_qtable__deps: ['$NNState'],
  rl_create_qtable: function (states, actions) {
    return NNState.module._rl_create_qtable(states, actions);
  },

  rl_free_qtable__sig: 'vi',
  rl_free_qtable__deps: ['$NNState'],
  rl_free_qtable: function (qtable) {
    NNState.module._rl_free_qtable(qtable);
  },

  rl_get_qtable__sig: 'fiii',
  rl_get_qtable__deps: ['$NNState'],
  rl_get_qtable: function (qtable, state, action) {
    return NNState.module._rl_get_qtable(qtable, state, action);
  },

  rl_set_qtable__sig: 'viiii',
  rl_set_qtable__deps: ['$NNState'],
  rl_set_qtable: function (qtable, state, action, value) {
    NNState.module._rl_set_qtable(qtable, state, action, value);
  },

  rl_get_qtable_states__sig: 'ii',
  rl_get_qtable_states__deps: ['$NNState'],
  rl_get_qtable_states: function (qtable) {
    return NNState.module._rl_get_qtable_states(qtable);
  },

  rl_get_qtable_actions__sig: 'ii',
  rl_get_qtable_actions__deps: ['$NNState'],
  rl_get_qtable_actions: function (qtable) {
    return NNState.module._rl_get_qtable_actions(qtable);
  },

  rl_qtable_save__sig: 'iii',
  rl_qtable_save__deps: ['$NNState'],
  rl_qtable_save: function (qtable, path) {
    return NNState.module._rl_qtable_save(qtable, path) ? 1 : 0;
  },

  rl_qtable_load__sig: 'ii',
  rl_qtable_load__deps: ['$NNState'],
  rl_qtable_load: function (path) {
    return NNState.module._rl_qtable_load(path);
  },

  // RL - Agent
  rl_create_agent__sig: 'iiiiff',
  rl_create_agent__deps: ['$NNState'],
  rl_create_agent: function (states, actions, lr, gamma) {
    return NNState.module._rl_create_agent(states, actions, lr, gamma);
  },

  rl_free_agent__sig: 'vi',
  rl_free_agent__deps: ['$NNState'],
  rl_free_agent: function (agent) {
    NNState.module._rl_free_agent(agent);
  },

  rl_choose_agent_action__sig: 'iii',
  rl_choose_agent_action__deps: ['$NNState'],
  rl_choose_agent_action: function (agent, state) {
    return NNState.module._rl_choose_agent_action(agent, state);
  },

  rl_update_agent__sig: 'viiifii',
  rl_update_agent__deps: ['$NNState'],
  rl_update_agent: function (agent, state, action, reward, next_state, done) {
    NNState.module._rl_update_agent(agent, state, action, reward, next_state, done);
  },

  rl_set_agent_policy__sig: 'viif',
  rl_set_agent_policy__deps: ['$NNState'],
  rl_set_agent_policy: function (agent, policy_type, epsilon) {
    NNState.module._rl_set_agent_policy(agent, policy_type, epsilon);
  },

  rl_get_agent_learning_rate__sig: 'fi',
  rl_get_agent_learning_rate__deps: ['$NNState'],
  rl_get_agent_learning_rate: function (agent) {
    return NNState.module._rl_get_agent_learning_rate(agent);
  },

  rl_set_agent_learning_rate__sig: 'vif',
  rl_set_agent_learning_rate__deps: ['$NNState'],
  rl_set_agent_learning_rate: function (agent, lr) {
    NNState.module._rl_set_agent_learning_rate(agent, lr);
  },

  rl_get_agent_discount_factor__sig: 'fi',
  rl_get_agent_discount_factor__deps: ['$NNState'],
  rl_get_agent_discount_factor: function (agent) {
    return NNState.module._rl_get_agent_discount_factor(agent);
  },

  rl_set_agent_discount_factor__sig: 'vif',
  rl_set_agent_discount_factor__deps: ['$NNState'],
  rl_set_agent_discount_factor: function (agent, gamma) {
    NNState.module._rl_set_agent_discount_factor(agent, gamma);
  },

  rl_get_agent_qtable__sig: 'ii',
  rl_get_agent_qtable__deps: ['$NNState'],
  rl_get_agent_qtable: function (agent) {
    return NNState.module._rl_get_agent_qtable(agent);
  },

  rl_set_agent_reward_clip__sig: 'viiff',
  rl_set_agent_reward_clip__deps: ['$NNState'],
  rl_set_agent_reward_clip: function (agent, enabled, min_val, max_val) {
    NNState.module._rl_set_agent_reward_clip(agent, enabled, min_val, max_val);
  },

  rl_set_agent_reward_normalization__sig: 'viif',
  rl_set_agent_reward_normalization__deps: ['$NNState'],
  rl_set_agent_reward_normalization: function (agent, enabled, scale) {
    NNState.module._rl_set_agent_reward_normalization(agent, enabled, scale);
  },

  rl_get_agent_average_reward__sig: 'di',
  rl_get_agent_average_reward__deps: ['$NNState'],
  rl_get_agent_average_reward: function (agent) {
    return NNState.module._rl_get_agent_average_reward(agent);
  },

  rl_get_agent_last_reward__sig: 'di',
  rl_get_agent_last_reward__deps: ['$NNState'],
  rl_get_agent_last_reward: function (agent) {
    return NNState.module._rl_get_agent_last_reward(agent);
  },

  rl_get_agent_episode_count__sig: 'ii',
  rl_get_agent_episode_count__deps: ['$NNState'],
  rl_get_agent_episode_count: function (agent) {
    return NNState.module._rl_get_agent_episode_count(agent);
  },

  rl_get_agent_last_episode_length__sig: 'ii',
  rl_get_agent_last_episode_length__deps: ['$NNState'],
  rl_get_agent_last_episode_length: function (agent) {
    return NNState.module._rl_get_agent_last_episode_length(agent);
  },

  rl_get_agent_average_episode_length__sig: 'di',
  rl_get_agent_average_episode_length__deps: ['$NNState'],
  rl_get_agent_average_episode_length: function (agent) {
    return NNState.module._rl_get_agent_average_episode_length(agent);
  },

  rl_notify_agent_episode_end__sig: 'vi',
  rl_notify_agent_episode_end__deps: ['$NNState'],
  rl_notify_agent_episode_end: function (agent) {
    NNState.module._rl_notify_agent_episode_end(agent);
  },

  rl_set_agent_epsilon_decay__sig: 'iidddii',
  rl_set_agent_epsilon_decay__deps: ['$NNState'],
  rl_set_agent_epsilon_decay: function (agent, start, min, rate, type, per_step) {
    return NNState.module._rl_set_agent_epsilon_decay(agent, start, min, rate, type, per_step) ? 1 : 0;
  },

  rl_update_agent_epsilon_step__sig: 'vi',
  rl_update_agent_epsilon_step__deps: ['$NNState'],
  rl_update_agent_epsilon_step: function (agent) {
    NNState.module._rl_update_agent_epsilon_step(agent);
  },

  rl_update_agent_epsilon_episode__sig: 'vi',
  rl_update_agent_epsilon_episode__deps: ['$NNState'],
  rl_update_agent_epsilon_episode: function (agent) {
    NNState.module._rl_update_agent_epsilon_episode(agent);
  },

  rl_get_agent_epsilon__sig: 'di',
  rl_get_agent_epsilon__deps: ['$NNState'],
  rl_get_agent_epsilon: function (agent) {
    return NNState.module._rl_get_agent_epsilon(agent);
  },

  rl_reset_agent_epsilon__sig: 'vi',
  rl_reset_agent_epsilon__deps: ['$NNState'],
  rl_reset_agent_epsilon: function (agent) {
    NNState.module._rl_reset_agent_epsilon(agent);
  },

  rl_set_agent_training__sig: 'vii',
  rl_set_agent_training__deps: ['$NNState'],
  rl_set_agent_training: function (agent, training) {
    NNState.module._rl_set_agent_training(agent, training);
  },

  rl_get_agent_training__sig: 'ii',
  rl_get_agent_training__deps: ['$NNState'],
  rl_get_agent_training: function (agent) {
    return NNState.module._rl_get_agent_training(agent);
  },

  rl_save_agent__sig: 'iii',
  rl_save_agent__deps: ['$NNState'],
  rl_save_agent: function (agent, path) {
    return NNState.module._rl_save_agent(agent, path) ? 1 : 0;
  },

  rl_load_agent__sig: 'ii',
  rl_load_agent__deps: ['$NNState'],
  rl_load_agent: function (path) {
    return NNState.module._rl_load_agent(path);
  },

});