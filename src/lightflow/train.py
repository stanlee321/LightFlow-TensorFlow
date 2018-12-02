from ..dataloader import load_batch
from ..dataset_configs import FLYING_CHAIRS_DATASET_CONFIG
from ..training_schedules import LONG_SCHEDULE
from .lightflow_tf import LightFlow

# Create a new network
net = LightFlow()

# Load a batch of data
input_a, input_b, flow = load_batch(FLYING_CHAIRS_DATASET_CONFIG, 'sample', net.global_step)
#
# Train on the data
net.train(
    log_dir='./logs/lightflow',
    training_schedule=LONG_SCHEDULE,
    input_a=input_a,
    input_b=input_b,
    flow=flow,
    # Load trained weights for CSS and SD parts of network
    #checkpoints={
        #'./checkpoints/FlowNetCSS-ft-sd/flownet-CSS-ft-sd.ckpt-0': ('FlowNet2/FlowNetCSS', 'FlowNet2'),
        #'./checkpoints/FlowNetSD/flownet-SD.ckpt-0': ('FlowNet2/FlowNetSD', 'FlowNet2')
    #}
    #checkpoints = {
    #    '../../logs/lightflow/model.ckpt-9468': ('lightflow/lightflow', 'lightflow')
    #}
    checkpoints = 'latest'
)