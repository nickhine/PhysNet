import tensorflow.compat.v1 as tf
from .layers.RBFLayer import *
from .layers.InteractionBlock import *
from .layers.OutputBlock      import *
from .activation_fn import *
from .grimme_d3.grimme_d3 import *

def softplus_inverse(x):
    '''numerically stable inverse of softplus transform'''
    return x + np.log(-np.expm1(-x))

class NeuralNetwork:
    def __str__(self):
        return "Neural Network"

    def __init__(self,
                 F,                              #dimensionality of feature vector
                 K,                              #number of radial basis functions
                 sr_cut,                         #cutoff distance for short range interactions
                 lr_cut = None,                  #cutoff distance for long range interactions (default: no cutoff)
                 num_blocks=3,                   #number of building blocks to be stacked
                 num_residual_atomic=2,          #number of residual layers for atomic refinements of feature vector
                 num_residual_interaction=2,     #number of residual layers for refinement of message vector
                 num_residual_output=1,          #number of residual layers for the output blocks
                 use_electrostatic=True,         #adds electrostatic contributions to atomic energy
                 use_ewald=False,                #periodic ewald interaction
                 use_dispersion=True,            #adds dispersion contributions to atomic energy
                 s6=None,                        #s6 coefficient for d3 dispersion, by default is learned
                 s8=None,                        #s8 coefficient for d3 dispersion, by default is learned
                 a1=None,                        #a1 coefficient for d3 dispersion, by default is learned
                 a2=None,                        #a2 coefficient for d3 dispersion, by default is learned   
                 Eshift=0.0,                     #initial value for output energy shift (makes convergence faster)
                 Escale=1.0,                     #initial value for output energy scale (makes convergence faster)
                 Qshift=0.0,                     #initial value for output charge shift 
                 Qscale=1.0,                     #initial value for output charge scale 
                 kehalf=7.199822675975274,       #half (else double counting) of the Coulomb constant (default is in units e=1, eV=1, A=1)
                 activation_fn=shifted_softplus, #activation function
                 dtype=tf.float32,               #single or double precision
                 seed=None,
                 scope=None):
        assert(num_blocks > 0)
        self._num_blocks = num_blocks
        self._dtype = dtype
        self._kehalf = kehalf
        self._F = F
        self._K = K
        self._sr_cut = sr_cut #cutoff for neural network interactions
        self._lr_cut = lr_cut #cutoff for long-range interactions
        self._use_electrostatic = use_electrostatic
        self._use_ewald = use_ewald
        self._use_dispersion = use_dispersion
        self._activation_fn = activation_fn
        self._scope = scope

        with tf.variable_scope(self.scope):
            #keep probability for dropout regularization
            self._keep_prob = tf.placeholder_with_default(1.0, shape=[], name="keep_prob")

            #atom embeddings (we go up to Pu(94), 95 because indices start with 0)
            self._embeddings = tf.Variable(tf.random_uniform([95, self.F], minval=-np.sqrt(3), maxval=np.sqrt(3), seed=seed, dtype=dtype), name="embeddings", dtype=dtype)
            tf.summary.histogram("embeddings", self.embeddings)  

            #radial basis function expansion layer
            self._rbf_layer = RBFLayer(K, sr_cut, scope="rbf_layer")

            #initialize variables for d3 dispersion (the way this is done, positive values are guaranteed)
            if s6 is None:
                self._s6 = tf.nn.softplus(tf.Variable(softplus_inverse(d3_s6), name="s6", dtype=dtype, trainable=True))
            else:
                self._s6 = tf.Variable(s6, name="s6", dtype=dtype, trainable=False)
            tf.summary.scalar("d3-s6", self.s6)
            if s8 is None:
                self._s8 = tf.nn.softplus(tf.Variable(softplus_inverse(d3_s8), name="s8", dtype=dtype, trainable=True))
            else:
                self._s8 = tf.Variable(s8, name="s8", dtype=dtype, trainable=False)
            tf.summary.scalar("d3-s8", self.s8)
            if a1 is None:
                self._a1 = tf.nn.softplus(tf.Variable(softplus_inverse(d3_a1), name="a1", dtype=dtype, trainable=True))
            else:
                self._a1 = tf.Variable(a1, name="a1", dtype=dtype, trainable=False)
            tf.summary.scalar("d3-a1", self.a1)
            if a2 is None:
                self._a2 = tf.nn.softplus(tf.Variable(softplus_inverse(d3_a2), name="a2", dtype=dtype, trainable=True))
            else:
                self._a2 = tf.Variable(a2, name="a2", dtype=dtype, trainable=False)
            tf.summary.scalar("d3-a2", self.a2)

            #initialize output scale/shift variables
            self._Eshift = tf.Variable(tf.constant(Eshift, shape=[95], dtype=dtype), name="Eshift", dtype=dtype)
            self._Escale = tf.Variable(tf.constant(Escale, shape=[95], dtype=dtype), name="Escale", dtype=dtype)
            self._Qshift = tf.Variable(tf.constant(Qshift, shape=[95], dtype=dtype), name="Qshift", dtype=dtype)
            self._Qscale = tf.Variable(tf.constant(Qscale, shape=[95], dtype=dtype), name="Qscale", dtype=dtype)

            #embedding blocks and output layers
            self._interaction_block = []
            self._output_block = []
            for i in range(num_blocks):
                self.interaction_block.append(
                    InteractionBlock(K, F, num_residual_atomic, num_residual_interaction, activation_fn=activation_fn, seed=seed, scope="interaction_block"+str(i), keep_prob=self.keep_prob, dtype=dtype))
                self.output_block.append(
                    OutputBlock(F, num_residual_output, activation_fn=activation_fn, seed=seed, scope="output_block"+str(i), keep_prob=self.keep_prob, dtype=dtype))
                                
            #saver node to save/restore the model
            self._saver = tf.train.Saver(self.variables, save_relative_paths=True, max_to_keep=50)

    def calculate_interatomic_distances(self, R, idx_i, idx_j, offsets=None):
        #calculate interatomic distances
        Ri = tf.gather(R, idx_i)
        Rj = tf.gather(R, idx_j)
        if offsets is not None:
            Rj += offsets
        Dij = tf.sqrt(tf.nn.relu(tf.reduce_sum((Ri-Rj)**2, -1))) #relu prevents negative numbers in sqrt
        return Dij

    #def calculate_interatomic_vectors(self, R, idx_i, idx_j, offsets=None):
    #    #calculate interatomic distances
    #    Ri = tf.gather(R, idx_i)
    #    Rj = tf.gather(R, idx_j)
    #    if offsets is not None:
    #        Rj += offsets
    #    Dij_vec = tf.convert_to_tensor(Ri-Rj)
    #    return Dij_vec

    #calculates the atomic energies, charges and distances (needed if unscaled charges are wanted e.g. for loss function)
    def atomic_properties(self, Z, R, idx_i, idx_j, offsets=None, sr_idx_i=None, sr_idx_j=None, sr_offsets=None):
        with tf.name_scope("atomic_properties"):
            #calculate distances (for long range interaction)
            Dij_lr = self.calculate_interatomic_distances(R, idx_i, idx_j, offsets=offsets)
            #optionally, it is possible to calculate separate distances for short range interactions (computational efficiency)
            if sr_idx_i is not None and sr_idx_j is not None:
                Dij_sr = self.calculate_interatomic_distances(R, sr_idx_i, sr_idx_j, offsets=sr_offsets)
            else:
                sr_idx_i = idx_i
                sr_idx_j = idx_j
                Dij_sr = Dij_lr
            #may need Dij as vectors if virial is being calculated
            #if True:
            #    Dij_vec = self.calculate_interatomic_vectors(R, idx_i, idx_j, offsets=offsets)
            #else:
            #    Dij_vec = None

            #calculate radial basis function expansion
            rbf = self.rbf_layer(Dij_sr)

            #initialize feature vectors according to embeddings for nuclear charges
            x = tf.gather(self.embeddings, Z)

            #apply blocks
            Ea = 0 #atomic energy 
            Qa = 0 #atomic charge
            nhloss = 0 #non-hierarchicality loss
            for i in range(self.num_blocks):
                x = self.interaction_block[i](x, rbf, sr_idx_i, sr_idx_j)
                out = self.output_block[i](x)
                Ea += out[:,0]
                Qa += out[:,1]
                #compute non-hierarchicality loss
                out2 = out**2
                if i > 0:
                    nhloss += tf.reduce_mean(out2/(out2 + lastout2 + 1e-7))
                lastout2 = out2

            #apply scaling/shifting
            Ea = tf.gather(self.Escale, Z) * Ea + tf.gather(self.Eshift, Z) + 0*tf.reduce_sum(R, -1) #last term necessary to guarantee no "None" in force evaluation
            Qa = tf.gather(self.Qscale, Z) * Qa + tf.gather(self.Qshift, Z)
        return Ea, Qa, Dij_lr, nhloss

    #calculates the energy given the scaled atomic properties (in order to prevent recomputation if atomic properties are calculated)
    def energy_from_scaled_atomic_properties(self, Ea, Qa, Dij, Z, R, idx_i, idx_j, cell=None, batch_seg=None):
        with tf.name_scope("energy_from_atomic_properties"):
            if batch_seg is None:
                batch_seg = tf.zeros_like(Z)
            #add electrostatic and dispersion contribution to atomic energy
            if self.use_electrostatic:
                Ea += self.electrostatic_energy_per_atom(Dij, Qa, R, idx_i, idx_j, cell)
            if self.use_dispersion:
                if self.lr_cut is not None:   
                    Ea += d3_autoev*edisp(Z, Dij/d3_autoang, idx_i, idx_j, s6=self.s6, s8=self.s8, a1=self.a1, a2=self.a2, cutoff=self.lr_cut/d3_autoang)
                else:
                    Ea += d3_autoev*edisp(Z, Dij/d3_autoang, idx_i, idx_j, s6=self.s6, s8=self.s8, a1=self.a1, a2=self.a2)
        return tf.squeeze(tf.segment_sum(Ea, batch_seg))

    #calculates the energy and forces given the scaled atomic atomic properties (in order to prevent recomputation if atomic properties are calculated)
    def energy_and_forces_from_scaled_atomic_properties(self, Ea, Qa, Dij, Z, R, idx_i, idx_j, cell, batch_seg=None):
        with tf.name_scope("energy_and_forces_from_atomic_properties"):
            energy = self.energy_from_scaled_atomic_properties(Ea, Qa, Dij, Z, R, idx_i, idx_j, cell, batch_seg)
            forces = -tf.convert_to_tensor(tf.gradients(tf.reduce_sum(energy), R)[0])
        return energy, forces

    #calculates the energy given the atomic properties (in order to prevent recomputation if atomic properties are calculated)
    def energy_from_atomic_properties(self, Ea, Qa, Dij, Z, idx_i, idx_j, cell, Q_tot=None, batch_seg=None):
        with tf.name_scope("energy_from_atomic_properties"):
            if batch_seg is None:
                batch_seg = tf.zeros_like(Z)
            #scale charges such that they have the desired total charge
            Qa = self.scaled_charges(Z, Qa, Q_tot, batch_seg)
        return self.energy_from_scaled_atomic_properties(Ea, Qa, Dij, Z, R, idx_i, idx_j, cell, batch_seg)

    #calculates the energy and force given the atomic properties (in order to prevent recomputation if atomic properties are calculated)
    def energy_and_forces_from_atomic_properties(self, Ea, Qa, Dij, Z, R, idx_i, idx_j, cell, Q_tot=None, batch_seg=None):
        with tf.name_scope("energy_and_forces_from_atomic_properties"):
            energy = self.energy_from_atomic_properties(Ea, Qa, Dij, Z, idx_i, idx_j, cell, Q_tot, batch_seg)
            forces = -tf.convert_to_tensor(tf.gradients(tf.reduce_sum(energy), R)[0])
        return energy, forces

    #calculates the total energy (including electrostatic interactions)
    def energy(self, Z, R, idx_i, idx_j, cell, Q_tot=None, batch_seg=None, offsets=None, sr_idx_i=None, sr_idx_j=None, sr_offsets=None):
        with tf.name_scope("energy"):
            Ea, Qa, Dij, _ = self.atomic_properties(Z, R, idx_i, idx_j, offsets, sr_idx_i, sr_idx_j, sr_offsets)
            energy = self.energy_from_atomic_properties(Ea, Qa, Dij, Z, idx_i, idx_j, cell, Q_tot, batch_seg)
        return energy 

    #calculates the total energy and forces (including electrostatic interactions)
    def energy_and_forces(self, Z, R, idx_i, idx_j, cell, Q_tot=None, batch_seg=None, offsets=None, sr_idx_i=None, sr_idx_j=None, sr_offsets=None):
        with tf.name_scope("energy_and_forces"):
            Ea, Qa, Dij, _ = self.atomic_properties(Z, R, idx_i, idx_j, offsets, sr_idx_i, sr_idx_j, sr_offsets)
            energy, forces = self.energy_and_forces_from_atomic_properties(Ea, Qa, Dij, Z, R, idx_i, idx_j, Q_tot, batch_seg)
        return energy, forces

    #returns scaled charges such that the sum of the partial atomic charges equals Q_tot (defaults to 0)
    def scaled_charges(self, Z, Qa, Q_tot=None, batch_seg=None):
        with tf.name_scope("scaled_charges"):
            if batch_seg is None:
                batch_seg = tf.zeros_like(Z)
            #number of atoms per batch (needed for charge scaling)
            Na_per_batch = tf.segment_sum(tf.ones_like(batch_seg, dtype=self.dtype), batch_seg)
            if Q_tot is None: #assume desired total charge zero if not given
                Q_tot = tf.zeros_like(Na_per_batch, dtype=self.dtype)
            #return scaled charges (such that they have the desired total charge)
            return Qa + tf.gather(((Q_tot-tf.segment_sum(Qa, batch_seg))/Na_per_batch), batch_seg)

    #switch function for electrostatic interaction (switches between shielded and unshielded electrostatic interaction)
    def _switch(self, Dij):
        cut = self.sr_cut/2
        x  = Dij/cut
        x3 = x*x*x
        x4 = x3*x
        x5 = x4*x
        return tf.where(Dij < cut, 6*x5-15*x4+10*x3, tf.ones_like(Dij))

    #calculates the electrostatic energy per atom 
    #for very small distances, the 1/r law is shielded to avoid singularities
    def electrostatic_energy_per_atom(self, Dij, Qa, R, idx_i, idx_j, cell):
        #gather charges
        Qi = tf.gather(Qa, idx_i)
        Qj = tf.gather(Qa, idx_j)
        #calculate variants of Dij which we need to calculate
        #the various shielded/non-shielded potentials
        DijS = tf.sqrt(Dij*Dij + 1.0) #shielded distance
        #calculate value of switching function
        switch = self._switch(Dij) #normal switch
        cswitch = 1.0-switch #complementary switch
        #calculate shielded/non-shielded potentials
        if self.lr_cut is None: #no non-bonded cutoff
            Eele_ordinary = 1.0/Dij   #ordinary electrostatic energy
            Eele_shielded = 1.0/DijS  #shielded electrostatic energy
            #combine shielded and ordinary interactions and apply prefactors
            if True:
                Eele = self.kehalf*Qi*Qj*(cswitch*Eele_shielded + switch*Eele_ordinary)
            else:
                Eele = self.kehalf*Qi*Qj*Eele_ordinary
            Eele_at = tf.segment_sum(Eele,idx_i)
        else: #with non-bonded cutoff
            cut   = self.lr_cut
            cut2  = self.lr_cut*self.lr_cut
            if not self.use_ewald:
                Eele_ordinary = 1.0/Dij  +  Dij/cut2 - 2.0/cut
                Eele_shielded = 1.0/DijS + DijS/cut2 - 2.0/cut
                #combine shielded and ordinary interactions and apply prefactors
                Eele = self.kehalf*Qi*Qj*(cswitch*Eele_shielded + switch*Eele_ordinary)
            else:
                Eele_ew_direct = tf.math.erfc(self.ewald_alpha*Dij)/Dij 
                Eele_ordinary = 1.0/Dij
                Eele_shielded = 1.0/DijS
                #combine ewald real space term with shielded and ordinary 
                #interactions and apply prefactors
                if True: # if True, apply shielded coulomb interaction
                   Eele = self.kehalf*Qi*Qj*(cswitch*(Eele_shielded - Eele_ordinary) + Eele_ew_direct)
                else:
                   Eele = self.kehalf*Qi*Qj*Eele_ew_direct # just the full Ewald term, no shielding
            Eele = tf.where(Dij <= cut, Eele, tf.zeros_like(Eele))
            Eele_at = tf.segment_sum(Eele,idx_i)
            if self.use_ewald:
                Eele_at = Eele_at + self.ewald_recip(Qa,R,cell) + self.ewald_self(Qa)
        return Eele_at 

    # Adapted from PyTorch code in SpookyNet by O. Unke.
    def ewald_recip(self,q,R,cell,eps=1e-8):
         # calculate reciprocal lattice vectors vectors
         k = self.get_kvecs(cell)
         # gaussian charge density
         k2 = tf.reduce_sum(k * k, -1)  # squared length of k-vectors
         qg = tf.math.exp(-0.25 * k2 / self.ewald_alpha2) / k2
         # fourier charge density
         dot = tf.reduce_sum(tf.expand_dims(R,-2) * k,-1) 
         q_real = tf.linalg.matvec(tf.math.cos(dot),q,transpose_a=True)
         q_imag = tf.linalg.matvec(tf.math.sin(dot),q,transpose_a=True)
         qf = q_real ** 2 + q_imag ** 2
         qf_sum = tf.reduce_sum(qf*qg,-1)
         two_pi_over_V = tf.reduce_sum(tf.cross(cell[0],cell[1])*cell[2],-1) * (self.two_pi)
         # reciprocal energy
         e_reciprocal = two_pi_over_V * qf_sum
         # spread reciprocal energy over atoms (to get an atomic contributions)
         q2 = q * q
         w = q2 + eps  # epsilon is added to prevent division by zero
         wnorm = tf.reduce_sum(w,-1)
         w = w / wnorm
         e_reciprocal = w * e_reciprocal
         return 2.0*self.kehalf*e_reciprocal 

    def ewald_self(self,q):
         # self interaction correction
         q2 = q * q
         e_self = -self.ewald_alpha * self.one_over_sqrtpi * q2
         return 2.0*self.kehalf*e_self

    def get_kvecs(self, cell):
        """ Generate reciprocal lattice vectors up to kmax for Ewald summation """
        import math
        kx = tf.range(0, self.ewald_Nmax[0] + 1, 1, self.dtype)
        kx = tf.concat([kx, -kx[1:]],axis=0)
        ky = tf.range(0, self.ewald_Nmax[1] + 1, 1, self.dtype)
        ky = tf.concat([ky, -ky[1:]],axis=0)
        kz = tf.range(0, self.ewald_Nmax[2] + 1, 1, self.dtype)
        kz = tf.concat([kz, -kz[1:]],axis=0)
        kvs = tf.stack(tf.meshgrid(kx, ky, kz, indexing='ij'), axis=-1)
        kvf = tf.reshape(kvs, (-1, 3))[1:]
        k = tf.math.scalar_mul(2.0*math.pi, tf.linalg.matvec(cell,kvf))
        kvi = tf.reduce_sum(tf.where(tf.math.reduce_sum(k**2,-1)<=self.ewald_kmax**2),-1)
        kvecs = tf.gather(k,kvi)
        return kvecs

    def set_ewald_params(self, alpha=None, kmax=None, Nmax=None):
        import math
        """ Set real space damping parameter for Ewald summation """
        if alpha is None:  # automatically determine alpha
            alpha = 4.0 / self.lr_cut + 1e-3
        self.ewald_alpha = alpha
        self.ewald_alpha2 = alpha ** 2
        self.two_pi = 2.0 * math.pi
        self.one_over_sqrtpi = 1 / math.sqrt(math.pi)
        # print a warning if alpha is so small that the reciprocal space sum
        # might "leak" into the damped part of the real space coulomb interaction
        if alpha * self.lr_cut < 4.0:  # erfc(4.0) ~ 1e-8
            print(f"Warning: Damping parameter alpha is {alpha} but probably should be at least {4.0 / self.lr_cut}")
        if kmax is None:
            self.ewald_kmax = 10
        else:
            self.ewald_kmax = kmax
        if Nmax is None:
            self.ewald_Nmax = [16,16,16]
        else:
            self.ewald_Nmax = Nmax

    #save the current model
    def save(self, sess, save_path, global_step=None):
        self.saver.save(sess, save_path, global_step)

    #load a model
    def restore(self, sess, save_path):
        self.saver.restore(sess, save_path)

    @property
    def keep_prob(self):
        return self._keep_prob
    
    @property
    def num_blocks(self):
        return self._num_blocks

    @property
    def dtype(self):
        return self._dtype

    @property
    def saver(self):
        return self._saver

    @property
    def scope(self):
        return self._scope
    
    @property
    def variables(self):
        scope_filter = self.scope + '/'
        varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_filter)
        return { v.name[len(scope_filter):]: v for v in varlist }
    
    @property
    def embeddings(self):
        return self._embeddings

    @property
    def Eshift(self):
        return self._Eshift

    @property
    def Escale(self):
        return self._Escale
  
    @property
    def Qshift(self):
        return self._Qshift

    @property
    def Qscale(self):
        return self._Qscale

    @property
    def s6(self):
        return self._s6

    @property
    def s8(self):
        return self._s8
    
    @property
    def a1(self):
        return self._a1

    @property
    def a2(self):
        return self._a2

    @property
    def use_electrostatic(self):
        return self._use_electrostatic

    @property
    def use_ewald(self):
        return self._use_ewald

    @property
    def use_dispersion(self):
        return self._use_dispersion

    @property
    def kehalf(self):
        return self._kehalf

    @property
    def F(self):
        return self._F

    @property
    def K(self):
        return self._K

    @property
    def sr_cut(self):
        return self._sr_cut

    @property
    def lr_cut(self):
        return self._lr_cut

    @property
    def activation_fn(self):
        return self._activation_fn
    
    @property
    def rbf_layer(self):
        return self._rbf_layer

    @property
    def interaction_block(self):
        return self._interaction_block

    @property
    def output_block(self):
        return self._output_block

