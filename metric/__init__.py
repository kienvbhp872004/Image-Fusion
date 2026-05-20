from .mean_intensity import mean_intensity
from .variance import variance
from .average_gradient import average_gradient
from .entropy import entropy
from .fmi import fmi
from .ncie import ncie
from .Nabf import nabf
from .scd import scd

# Matlab / Standard counterparts
from .mutual_information import mutual_information
from .cross_entropy import cross_entropy
from .psnr import psnr
from .ssim import ssim
from .rmse import rmse
from .spatial_frequency import spatial_frequency
from .edge_intensity import edge_intensity

# Liu 2012 / VIFB metrics
from .gradient_preservation_qg import QG as qg_petrovic
from .wavelet_qm import QM as wavelet_qm
from .piella_qc import QC as piella_qc
from .piella_qs import QS as piella_qs
from .chen_blum_qcb import QCB as chen_blum
from .chen_varshney_qcv import QCV as chen_varshney
from .yang_qy import QY as yang_ssim
from .mi_normalized import QMI as mi_normalized
from .sf_relative import QSF as sf_relative
from .ncc_entropy_qncie import QNCIE as ncc_entropy
from .tsallis_entropy_qte import QTE as tsallis_entropy

# Optional
try:
    from .phase_congruency_qp import QP as phase_congruency
except ImportError:
    phase_congruency = None
