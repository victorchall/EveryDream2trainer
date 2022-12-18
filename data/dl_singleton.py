# stop lightning's repeated instantiation of batch train/val/test classes causing multiple sweeps of the same data off disk
shared_dataloader = None
