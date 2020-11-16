Examples
========

The default configuration is:

.. testsetup::

   from spest.config import Config

.. testcode::
   
   print(Config())

.. testoutput::

   builtins.Config:
                   batch_size: 32
         boundary_loss_weight: 1
           center_loss_weight: 1
                    eval_step: 100
              image_save_step: 100
              image_save_zoom: 1
              kernel_avg_beta: 0.99
                kernel_length: 21
              kn_num_channels: 1024
               kn_num_linears: 4
               kn_update_step: 1
              lrd_kernel_size: (3, 1)
             lrd_num_channels: 64
                lrd_num_convs: 5
              lrelu_neg_slope: 0.1
                   num_epochs: 10000
              num_init_epochs: 0
                   patch_size: 16
                 scale_factor: 1
       smoothness_loss_weight: 1
                 weight_decay: 0
