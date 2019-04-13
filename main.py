import network_trainer as nt
import acdc_data_loader as acdc

cropper_size = acdc.get_cropper_size()

# conv 2D default parameter: channels last: (batch, rows, cols, channels)
nt.network_trainer(cropper_size)

