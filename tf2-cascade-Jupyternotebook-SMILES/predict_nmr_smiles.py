import os
import pandas as pd
import numpy as np
from rdkit import Chem
from nfp.preprocessing import MolAPreprocessor, GraphSequence

import keras
import keras.backend as K

from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

from keras.layers import (
    Input,
    Embedding,
    Dense,
    BatchNormalization,
    Concatenate,
    Multiply,
    Add,
)

from keras.models import Model, load_model

from nfp.layers import (
    MessageLayer,
    GRUStep,
    Squeeze,
    EdgeNetwork,
    ReduceBondToPro,
    ReduceBondToAtom,
    GatherAtomToBond,
    ReduceAtomToPro,
)
from nfp.models import GraphModel
from cascade.apply import predict_NMR_C, predict_NMR_H


# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

modelpath_C = os.path.join("cascade", "trained_model", "best_model.hdf5")
modelpath_H = os.path.join("cascade", "trained_model", "best_model_H_DFTNN.hdf5")

batch_size = 32
atom_means = pd.Series(
    np.array([0, 0, 97.74193, 0, 0, 0, 0, 0, 0, 0]).astype(np.float64), name="shift"
)
NMR_model_C = load_model(
    modelpath_C,
    custom_objects={
        "GraphModel": GraphModel,
        "ReduceAtomToPro": ReduceAtomToPro,
        "Squeeze": Squeeze,
        "GatherAtomToBond": GatherAtomToBond,
        "ReduceBondToAtom": ReduceBondToAtom,
    },
)
NMR_model_H = load_model(
    modelpath_H,
    custom_objects={
        "GraphModel": GraphModel,
        "ReduceAtomToPro": ReduceAtomToPro,
        "Squeeze": Squeeze,
        "GatherAtomToBond": GatherAtomToBond,
        "ReduceBondToAtom": ReduceBondToAtom,
    },
)

data = [
    "CC(C(=O)C(C)NC(=O)C)(C)C1=CC=CC=C1C(=O)C2=CC=CC=C2",
    "CC1(C)C(O)C(O)C2=CC=CC(=C2)C3=CC=CC=C3C(C4=CC=CC=C4)N=C1C5=CC=CC=C5",
    "CCOC(=O)C(CCC1=CC=CC(=C1)[NH1]C2=CC=CC=C2)CCC=CC=COC3=C4C=CC=C3C=CC=CC=CC=C=C4[P+1]=[P+1]",
]  # dummy input


def predict_nmr_for_smiles(smiles_list):
    results = []

    for smiles in smiles_list:
        try:
            # Predict NMR for Carbon
            mols_C, weightedPrediction_C, spreadShift_C = predict_NMR_C(
                smiles, NMR_model_C
            )
            weightedPrediction_C["SMILES"] = smiles

            # Predict NMR for Hydrogen
            mols_H, weightedPrediction_H, spreadShift_H = predict_NMR_H(
                smiles, NMR_model_H
            )
            weightedPrediction_H["SMILES"] = smiles

            # Add tuple to results
            results.append((smiles, weightedPrediction_C, weightedPrediction_H))

        except Exception as e:
            print(f"Error processing {smiles}: {e}")
            results.append((smiles, None, None))

    return results
