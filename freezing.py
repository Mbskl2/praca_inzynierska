from keras.models import Sequential
import numpy as np

class FreezingSequential(Sequential):
    def __init__(self, percent_of_frozen_neurons):
        super().__init__()
        self.percent_of_frozen_neurons = percent_of_frozen_neurons

    def fit(self, x=None,
            y=None,
            batch_size=32,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            **kwargs):
            for i in range(epochs):
                print('Epoch #', i+1, sep='', end=' ')
                old_weights, masks = self.pick_old_weights_and_masks(self.get_weights())
                super().fit(x,y, batch_size,1,verbose,callbacks,validation_split,validation_data,shuffle,class_weight,sample_weight,initial_epoch)
                if i < epochs - 1:
                    self.set_weights(self.get_new_weights(self.get_weights(), old_weights, masks))
    
    def pick_old_weights_and_masks(self, weights):
        old_weights = []
        masks = []
        for matrix in weights:
            mask = np.random.rand(*matrix.shape) * 100 < self.percent_of_frozen_neurons
            old_weights.append(matrix[mask])
            masks.append(mask)
        return old_weights, masks

    def get_new_weights(self, current_weights, old_weights, masks):
        new_weights = []
        for current, old, mask in zip(current_weights, old_weights, masks):
            current[mask] = old
            new_weights.append(current) 
        return new_weights