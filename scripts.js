const configForm = document.getElementById('configForm');
const launchButton = document.getElementById('launchButton');

const mainConfig = {
  "batch_size": 10,
  "data_root": "X:\\my_project_data\\project_abc",
  "flip_p": 0.0,
  "gradient_checkpointing": true,
  "logdir": "logs",
  "lowvram": false,
  "lr": 1.5e-06,
  "lr_decay_steps": 0,
  "lr_scheduler": "constant",
  "lr_warmup_steps": null,
  "max_epochs": 30,
  "project_name": "project_abc",
  "resolution": 512,
  "resume_ckpt": "sd_v1-5_vae",
  "sample_prompts": "sample_prompts.json",
  "sample_steps": 300,
  "save_ckpt_dir": null,
  "save_every_n_epochs": 20,
  "seed": 555,
  "zero_frequency_noise_ratio": 0.02
};

const optionalConfig = {
  "shuffle_tags": false,
  "grad_accum": 1,
  "log_step": 25,
  "validation_config": null,
  "wandb": false,
  "write_schedule": false,
  "rated_dataset": false,
  "rated_dataset_target_dropout_percent": 50,
  "ckpt_every_n_minutes": null,
  "clip_grad_norm": null,
  "clip_skip": 0,
  "cond_dropout": 0.04,
  "disable_textenc_training": false,
  "disable_xformers": false,
  "gpuid": 0,
  
};

const config = { ...mainConfig, ...optionalConfig };

// Generate input fields for the main JSON config
for (const key in mainConfig) {
  addConfigField(key, mainConfig[key], configForm);
}

const toggleOptionalConfig = document.createElement('input');
toggleOptionalConfig.type = 'checkbox';
toggleOptionalConfig.id = 'toggleOptionalConfig';
toggleOptionalConfig.name = 'toggleOptionalConfig';
configForm.appendChild(toggleOptionalConfig);
const toggleOptionalConfigLabel = document.createElement('label');
toggleOptionalConfigLabel.htmlFor = 'toggleOptionalConfig';
toggleOptionalConfigLabel.innerText = 'Show advanced options';
configForm.appendChild(toggleOptionalConfigLabel);

const optionalConfigContainer = document.createElement('div');
optionalConfigContainer.id = 'optionalConfigContainer';
optionalConfigContainer.style.display = 'none';
configForm.appendChild(optionalConfigContainer);

// Generate input fields for the optional JSON config
for (const key in optionalConfig) {
  addConfigField(key, optionalConfig[key], optionalConfigContainer);
}

toggleOptionalConfig.addEventListener('change', () => {
  optionalConfigContainer.style.display = toggleOptionalConfig.checked ? 'block' : 'none';
});

launchButton.addEventListener('click', () => {
  launchButton.disabled = true;
  updateConfigFromForm();
  document.getElementById('configInput').value = JSON.stringify(config);
  document.getElementById('launchForm').submit();
});

function addConfigField(key, value, container) {
  const label = document.createElement('label');
  label.htmlFor = key;
  label.innerText = `${key}:`;

  const input = document.createElement('input');
  input.type = typeof value === 'boolean' ? 'checkbox' : 'text';
  input.id = key;
  input.name = key;
  input.value = value;
  if (typeof value === 'boolean') {
    input.checked = value;
  }

  container.appendChild(label);
  container.appendChild(input);
  container.appendChild(document.createElement('br'));
}

function updateConfigFromForm() {
  for (const key in config) {
    const input = document.getElementById(key);
    config[key] = typeof config[key] === 'boolean' ? input.checked : input.value;
  }
}
 
