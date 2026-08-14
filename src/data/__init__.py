"""Couche data engineering : ingestion, preprocessing, splits, garde-fous.

Ce package ne contient ni modele GAN, ni boucle d'entrainement : uniquement la
chaine qui prepare et alimente les donnees, plus la plomberie de l'evaluation.
"""

from src.data import config

__all__ = ["config"]
