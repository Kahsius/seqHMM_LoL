setwd("~/Dropbox/Thèse/R/") # Xubuntu
# setwd("../Dropbox/Thèse/R/") # Windows
library("seqHMM")
library("rjson")
source("./gen_prob_matrix.R")

# A partir d'ici, les informations utilisees sont generiques et pour pouvoir utiliser
# le systeme, il faut les donneees suivantes
# 
# matrice.global <- liste de matrices représentant chacun des canaux
#                |_ Une matrice est de taille n*m avec n le nombre de sequences et m la duree maximale d'une sequence (TMAX)
#                |_ Une matrice est organisée en ligne de la manière suivante par match, puis équipe, puis membre de l'équipe
# 
# vocab <- vector des noms donnees a chaque canal, dans le meme ordre que matrice.global
# alphabets <- vector des differents alphabets pour chaque canal, vecteurs de strings
#                  |_ a fournir au cas oe il y aurait des valeurs manquantes dans les donnees

print("Initialisation des paramètres")
load("./package_data_starcraft.RData")

len <- dim(matrice.global[[1]])[1]
n_matches <- len/N_P_PER_TEAM/N_TEAM
need_seq = T
if(exists("seqdef.mmm")) need_seq = F
# Extraction de la longueur maximale
l_max <- max(apply(matrice.global[[1]], 1, function(row){
  return(max(which(row!="NA")))
}))

N_STATES <-30
N_STATES2 <- 10
N_CLUSTERS_PROBAS <- 10
N_TEAM <- 1
N_P_PER_TEAM <- 1
TMAX <- 14 # <=  la duree minimum d'une partie pour le moment. <- l_max si pas de limitation
TRIANGLE <- F
N_THREADS <- 4

print("Definition des sequences pour l'apprentissage du futur mixture modele")
# Changement du format de donnees en sequences pour l'ensemble des donnees
if(need_seq){
  matrice.seqdef <- vector("list", length(vocab))
  for(i in 1:length(vocab)){
    matrice.seqdef[[i]] <- seqdef(matrice.global[[i]], alphabet=alphabets[[i]])
    attr(matrice.seqdef[[i]], "cpal") <- colorpalette[[length(alphabets[[i]])]]
  }
  
  print("Definition des sequences pour l'apprentissage par equipe")
  print("--- Recuperation des sequences dans la matrice globale")
  
  # ====--== DISTINCTION DES EQUIPES ========- #
  seq.each.team <- vector("list")
  seq.each.team$team <- vector("list", N_TEAM)
  # Initialisation de toutes les séquences de toutes les équipes
  for(i in 1:N_TEAM){
    seq.each.team$team[[i]] <- vector("list", length(vocab))
    for(j in 1:length(vocab)){
      seq.each.team$team[[i]][[j]] <- matrix(NA, nrow = len/N_TEAM, ncol = TMAX)
    }
  }
  for(i in 1:(n_matches)){
    for(j in 1:length(vocab)){
      for(k in 1:N_TEAM){
        # Désolé pour la lisibilité, j'ai fait au mieux
        range <- (i-1)*N_P_PER_TEAM
        range2 <- range*N_TEAM + (k-1)*N_P_PER_TEAM
        v <- matrice.global[[j]][(1+range2):(N_P_PER_TEAM+range2),]
        index <- min(TMAX, dim(v)[2])
        seq.each.team$team[[k]][[j]][(1+range):(N_P_PER_TEAM+range),1:index] <- v[,1:index]
      }
    }
  }
  
  
  print("--- Passage au format sequence")
  # L'objet seq.each.team contient tout ce qu'il faut pour chaque equipe, que ea soit les matrices de base ou les seqdef
  seq.each.team$sequences <- vector("list")
  seq.each.team$sequences$team <- vector("list", N_TEAM)
  for(i in 1:N_TEAM){
    seq.each.team$sequences$team[[i]] <- vector("list", length(vocab))
    for(j in 1:length(vocab)){
      seq.each.team$sequences$team[[i]][[j]] <- seqdef(seq.each.team$team[[i]][[j]], alphabet = alphabets[[j]])
      attr(seq.each.team$sequences$team[[i]][[j]], "cpal") <- colorpalette[length(alphabets[[j]])]
    }
  }  
}


# Visualisation des listes de la k eme equipe
# k = 1
# ssplot(seq.each.team$sequences$team[[k]], title="state distribution")

# Création des paramètres des HMMs
print("Creation des HMMs par equipe")
tpm <- gen_prob_matrix(N_STATES, N_STATES, triangle = TRIANGLE)
rho <- vector("list", length(vocab))
for(i in 1:length(vocab)){
  rho[[i]] <- gen_prob_matrix(N_STATES, length(alphabets[[i]]))
}
init <- runif(N_STATES)
init <- init/sum(init)

print("--- Initialisation")
# HMM avec positions + events
hmm.per.team <- vector("list")
hmm.per.team$hmms <- vector("list", N_TEAM)
hmm.per.team$fitted.hmms <- vector("list", N_TEAM)
for(i in 1:N_TEAM){
  hmm.per.team$hmms[[i]] <- build_hmm(
    observations = seq.each.team$sequences$team[[i]],
    initial_probs = init,
    emission_probs = rho,
    transition_probs = tpm,
    channel_names = vocab
  )
}

print("--- Apprentissage")
# HMM global (without consideration of team)
# fhmm <- fit_model(hmm, threads = 4)
# HMM for each team
for(i in 1:N_TEAM){
  hmm.per.team$fitted.hmms[[i]] <- fit_model(hmm.per.team$hmms[[i]], threads = N_THREADS)
}

print("Creation du mixture modele")
print("--- Recuperation des modeles pour chaque equipe")
# Mixture HMM, afin de representer l'ensemble de la partie
models <- vector("list", N_TEAM)
models$all_initial_probs <- vector("list", N_TEAM)
models$all_transition_probs <- vector("list", N_TEAM)
models$all_emission_probs <- vector("list", N_TEAM)
models$all_cluster_names <- vector(length = N_TEAM)

for(i in 1:N_TEAM){
  models[[i]] <- hmm.per.team$fitted.hmms[[i]]$model
  models$all_initial_probs[[i]] <- models[[i]]$initial_probs
  models$all_transition_probs[[i]] <- models[[i]]$transition_probs
  models$all_emission_probs[[i]] <- models[[i]]$emission_probs
  models$all_cluster_names[i] <- paste("Team",i)
}

print("--- Initialisation du modele")
mhmm <- build_mhmm(
  observations = matrice.seqdef,
  initial_probs = models$all_initial_probs,
  transition_probs = models$all_transition_probs,
  emission_probs = models$all_emission_probs,
  cluster_names = models$all_cluster_names,
  channel_names = vocab
)

print("--- Apprentissage du modele")
fmhmm <- fit_model(model = mhmm, threads = N_THREADS)

### Tentative de mega apprentissage
print("Creation de la couche superieure")
# Creation des donnees
# On recupere d'abord les probabilites de passage par les differents etats
# sur chaque séquence a l'aide des forwards
# a noter que vu que fmhmm est un mixture modele, il faudra a chaque fois
# faire attention aux etats selectionnes (*)
print("--- Recuperation des forwards probabilities")
r <- forward_backward(fmhmm$model, forward_only = TRUE, threads = N_THREADS)
m_forward <- vector("list", N_TEAM)
for(i in 1:N_TEAM){
  m_forward[[i]] <- vector("list", len/N_TEAM)
}
for(i in 1:n_matches){
  for(j in 1:N_P_PER_TEAM){
    for(k in 1:N_TEAM){
      m_forward[[k]][[(i-1)*N_P_PER_TEAM+j]] <- r$forward_probs[,,(i-1)*N_P_PER_TEAM*N_TEAM+j+(k-1)*N_P_PER_TEAM]
      m_forward[[k]][[(i-1)*N_P_PER_TEAM+j]] <- m_forward[[k]][[(i-1)*N_P_PER_TEAM+j]][1:N_STATES+(k-1)*N_STATES,] # (*)
    }
  }
}

print("--- Mise en forme et sommage")
m_all_forward <- vector("list", n_matches)
for(i in 1:n_matches){
  t <- matrix(data = 0, nrow = N_TEAM*N_STATES, ncol = TMAX)
  for(l in 1:TMAX){
    for(j in 1:N_TEAM){
      v <- vector("numeric", N_STATES)
      for(k in 1:N_P_PER_TEAM){
        v <- v + m_forward[[j]][[k+(i-1)*N_P_PER_TEAM]][,l]
      }
      t[1:N_STATES+(j-1)*N_STATES,l] <- v
    }    
  }
  colnames(t) <- colnames(r$forward_probs[,,1])
  rownames(t) <- rownames(r$forward_probs[,,1])
  m_all_forward[[i]] <- t
}

# Reorganisation suivants les canaux (= les etats caches des HMMs inferieurs)
# m_all_forward est organise suivant les matchs, et pas suivants les canaux
print("--- Reorganisation des donnees suivant les canaux")
mmm <- vector("list", N_TEAM*N_STATES)
mmm2 <- vector("list", N_TEAM*N_STATES)
for (i in 1:(N_TEAM*N_STATES)){
  mtmp <- matrix(0, n_matches, TMAX)
  rownames(mtmp) <- 1:n_matches
  colnames(mtmp) <- colnames(r$forward_probs[,,1])
  for (j in 1:n_matches){
    mtmp[j,] <- m_all_forward[[j]][i,]
  }
  mmm[[i]] <- mtmp
}

# Creation du clustering pour les probas
# L'idee, c'est de separer les probas (continues) en differents clusters (discrets)
# afin de pouvoir fournir des classes a emettre par le HMM
print("--- Definition du clustering des probabilites sur les etats caches")
mmsum <- m_all_forward[[1]]
for(i in 2:n_matches){
  mmsum <- mmsum+m_all_forward[[i]]
}
mmsum <- mmsum/n_matches
h <- hist(mmsum,breaks = prod(dim(mmsum)))
co <- h$counts
h2 <- hist(cumsum(co), breaks = 0:N_CLUSTERS_PROBAS*max(cumsum(co))/N_CLUSTERS_PROBAS)
co2 <- h2$counts
segments <- cumsum(co2)
segments_probas <- h$breaks[segments+1]

# Reformatage de mmm pour avoir les segments au lieu des probas
print("--- Reformatage des probabilites e l'aide des segments")
for(i in 1:(N_TEAM*N_STATES)){
  mtmp <- mmm[[i]]
  for(j in N_CLUSTERS_PROBAS:1){
    mtmp[mmm[[i]] <= segments_probas[j]] <- j
  }
  mmm2[[i]] <- mtmp
}

print("--- Definition des sequences pour l'apprentissage superieur")
seqdef.mmm <- vector("list", N_TEAM*N_STATES)
for(i in 1:(N_TEAM*N_STATES)){
  seqdef.mmm[[i]] <- seqdef(
    data = mmm2[[i]],
    alphabet = 1:N_CLUSTERS_PROBAS
  )
}

# Creation du mega hmm
print("--- Initialisation du HMM superieur")
tpm <- gen_prob_matrix(N_STATES2, N_STATES2, triangle = TRIANGLE)
rho <- vector("list", N_STATES*N_TEAM)
for(i in 1:(N_TEAM*N_STATES)){
  rho[[i]] <- gen_prob_matrix(N_STATES2, N_CLUSTERS_PROBAS)
}
init <- runif(N_STATES2)
init <- init/sum(init)

megahmm <- build_hmm(
  observations = seqdef.mmm,
  initial_probs = init,
  emission_probs = rho,
  transition_probs = tpm,
  channel_names = rownames(r$forward_probs[,,1])
)

print("--- Apprentissage du HMM superieur")
fmegahmm <- fit_model(megahmm, threads = N_THREADS)
# [OPTIONNEL] trim du model
model <- trim_model(fmegahmm$model, maxit=100, zerotol=9e-5)

# ====================================================-- #
# ============= Debuts de systeme d'analyse ============ #
# ====================================================-- #
# ======================== TODO ======================== #
# ====================================================-- #

# Analyse état part état
# Pour chaque état, on regarde les probabilités d'émission associées
# Pour avoir une idée de ce que l'état représente
# Pour l'instant on ne regarde que les etats binaires

# states <- vector("list", N_STATES2)
# for(i in 1:N_STATES2){
#   state <- matrix(0, 2*N_STATES, N_CLUSTERS_PROBAS)
#   for(j in 1:(2*N_STATES)){
#     state[j,] <- fmegahmm$model$emission_probs[[j]][i,]
#   }
#   rownames(state) <- names(fmegahmm$model$emission_probs)
#   states[[i]] <- state
# }
# 
states.t1 <- vector("list", N_STATES)
for(i in 1:N_STATES){
  state <- matrix(0,length(vocab),2)
  for(j in 1:length(vocab)){
    state[j,] <- fmhmm$model$emission_probs$`Team 1`[[j]][i,]
  }
  rownames(state) <- names(h$model$emission_probs)
  states.t1[[i]] <- state
}
# 
# states.t2 <- vector("list", N_STATES)
# for(i in 1:N_STATES){
#   state <- matrix(0,length(vocab),2)
#   for(j in 1:length(vocab)){
#     state[j,] <- fmhmm$model$emission_probs$`Team 2`[[j]][i,]
#   }
#   rownames(state) <- names(fhmm.t2$model$emission_probs)[1:13]
#   states.t2[[i]] <- state
# }
