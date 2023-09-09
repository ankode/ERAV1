# Transformer Model for English-French Translation

This repository contains code for training a Transformer model for English to French translation. The goal is to create a model that achieves a loss under 1.8 on a specific dataset.

## Dataset
The dataset used for this project is the "en-fr" dataset from Opus Books.

## Data Preprocessing
Before training the Transformer model, the following preprocessing steps were performed on the dataset:

1. **Removing Long English Sentences**: All English sentences with more than 150 tokens were removed from the dataset.

2. **Removing Corresponding French Sentences**: For each English sentence removed, the corresponding French sentence was also removed if its length exceeded the length of the English sentence plus 10 tokens.

## Transformer Model
The core of this project is the implementation of a Transformer model for English-French translation. The model includes an encoder-decoder architecture with parameter sharing (PS) and utilizes mixed-precision training with the Automatic Mixed Precision (AMP) library.

## Training Logs

Below is a summary of training progress for the last 5 epochs, including metrics such as loss, accuracy, or any other relevant information. This can help you monitor the model's performance during training.

**Epoch 15: train_loss=1.9422**
*****************************************
    SOURCE: It was the coin which the man in the black mantle had given to Phoebus.
    TARGET: C’était la pièce que l’homme au manteau noir avait donnée à Phœbus.
    PREDICTED: C ’ était la monnaie que l ’ homme au manteau noir avait donné à Phœbus .
*****************************************

*****************************************
    SOURCE: In class, Jasmin related his night adventure.
    TARGET: En classe, Jasmin raconta son aventure de la nuit :
    PREDICTED: Dans la classe , Jasmin raconta son aventure .
*****************************************

**Epoch 16: train_loss=2.2064**
*****************************************
    SOURCE: The conversation between the friends was endless; Korasoff was in raptures: never had a Frenchman given him so long a hearing.
    TARGET: La conversation entre les deux amis fut infinie ; Korasoff était ravi : jamais un Français ne l’avait écouté aussi longtemps.
    PREDICTED: La conversation entre les amis était infinie , c ’ était Korasoff : jamais un Français ne l ’ avait donné tant de longtemps .
*****************************************

*****************************************
    SOURCE: At the sixth he began to reflect that the search was rather dubious.
    TARGET: Au sixième, il commença de réfléchir que la recherche était un peu hasardée.
        PREDICTED: Au sixième , il se mit a réfléchir , que la recherche d ' un air assez louche .
*****************************************

**Epoch 17: train_loss=2.0881**
*****************************************
    SOURCE: The horror of death reanimated her,−−
    TARGET: L’horreur de la mort la ranima.
     PREDICTED: L ’ horreur de la mort lui :
*****************************************

*****************************************
    SOURCE: Going close by a bench on the breakwater he sat down, tired already ofwalking and out of humour with his stroll before he had taken it.
    TARGET: Comme il frôlait un banc sur le brise-lames, il s'assit, déjà las demarcher et dégoûté de sa promenade avant même de l'avoir faite.
    PREDICTED: Il se tenait à un banc , sur le brise - lames , déjà las et de l ' air d ' être mis à la promenade , avant qu ' il l ' eût .
*****************************************

**Epoch 18: train_loss=1.8016**
*****************************************
    SOURCE: A very good workman, he could speak well, put himself at the head of every opposition, and had at last become the chief of the discontented.
    TARGET: Tres bon ouvrier, il parlait bien, se mettait a la tete de toutes les réclamations, avait fini par etre le chef des mécontents.
    PREDICTED: Un bon ouvrier , on pouvait parler , se mettait a la tete , en tete de tous , en finit par faire de la .
*****************************************

*****************************************
    SOURCE: He redoubled his attention.
    TARGET: Il redoubla d’attention.
    PREDICTED: Il redoubla d ' attention .
*****************************************

**Epoch 19: train_loss=2.0021**
Validation DataLoader 0: 55%
4860/8844 [00:23<00:19, 205.28it/s]
*****************************************
    SOURCE: Sometimes bending over the forecastle railings, sometimes leaning against the sternrail, I eagerly scoured that cotton-colored wake that whitened the ocean as far as the eye could see!
    TARGET: Tantôt penché sur les bastingages du gaillard d'avant, tantôt appuyé à la lisse de l'arrière, je dévorais d'un oeil avide le cotonneux sillage qui blanchissait la mer jusqu'à perte de vue !
    PREDICTED: Parfois , par le gaillard des deux grilles , tantôt , je avec ardeur à l ' du haut que l ' Océan même pouvait voir !
*****************************************

*****************************************
    SOURCE: They had recognized Quasimodo.
    TARGET: On avait reconnu Quasimodo.
    PREDICTED: On avait reconnu Quasimodo .
*****************************************