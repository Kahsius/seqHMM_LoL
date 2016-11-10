setwd("~/Dropbox/Thèse/R/") # Xubuntu
# setwd("../Dropbox/Thèse/R/") # Windows
library("seqHMM")
library("rjson")
source("./gen_prob_matrix.R")
source("./myImagePlot.R")

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
load("./package_starcraft_pvp.RData")

len <- dim(matrice.global[[1]])[1]
need_seq = !exists("seqdef.mmm")
# Extraction de la longueur maximale
l_max <- max(apply(matrice.global[[1]], 1, function(row){
  return(max(which(row!="NA"))) # aussi sum(!is.na(row))
}))

N_STATES <-30
N_STATES2 <- 100
N_CLUSTERS_PROBAS <- 10
N_TEAM <- 2
N_P_PER_TEAM <- 1
TMAX <- l_max # <=  la duree minimum d'une partie pour le moment. <- l_max si pas de limitation
TRIANGLE <- F
N_THREADS <- 4
MODE_PER_TEAM <- F
SEUIL_REPRESENTATIVITE <- 0.01
n_matches <- len/N_P_PER_TEAM/N_TEAM

print("Definition des sequences pour l'apprentissage du futur mixture modele")
# Changement du format de donnees en sequences pour l'ensemble des donnees
if(need_seq){
  # matrice.seqdef <- vector("list", length(vocab))
  # for(i in 1:length(vocab)){
  #   matrice.seqdef[[i]] <- seqdef(matrice.global[[i]], alphabet=alphabets[[i]])
  #   attr(matrice.seqdef[[i]], "cpal") <- colorpalette[[length(alphabets[[i]])]]
  # }
  
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
        v <- matrice.global[[j]][1:N_P_PER_TEAM+range2,]
        index <- min(TMAX, sum(!is.na(v)))
        if(N_P_PER_TEAM>1){
          v <- v[,1:index]
        } else {
          v <- v[1:index]
        }
        seq.each.team$team[[k]][[j]][1:N_P_PER_TEAM+range,1:index] <- v
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
# HMM for each team
for(i in 1:N_TEAM){
  print(paste("    --- Learning HMM", i))
  hmm.per.team$fitted.hmms[[i]] <- fit_model(hmm.per.team$hmms[[i]], threads = N_THREADS)
  print("Done")
}

### Tentative de mega apprentissage
print("Creation de la couche superieure")
# Creation des donnees
# On recupere d'abord les probabilites de passage par les differents etats
# sur chaque séquence a l'aide des forwards
print("--- Recuperation des forwards probabilities")
r <- vector("list", N_TEAM)
for(i in 1:N_TEAM){
  r[[i]] <- forward_backward(hmm.per.team$fitted.hmms[[i]]$model, forward_only = T, threads = N_THREADS)
}
# m_forward[[équipe]][[séquence d'un joueur de cette team]]
m_forward <- vector("list", N_TEAM)
for(i in 1:N_TEAM){
  m_forward[[i]] <- vector("list", len/N_TEAM)
}
for(i in 1:n_matches){
  for(j in 1:N_P_PER_TEAM){
    for(k in 1:N_TEAM){
      m_forward[[k]][[(i-1)*N_P_PER_TEAM+j]] <- r[[k]]$forward_probs[,,(i-1)*N_P_PER_TEAM+j]
    }
  }
}

if(!MODE_PER_TEAM) {
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
    rn <- as.vector(matrix(NA, nrow = 1, ncol = N_TEAM*N_STATES))
    for(k in 1:N_TEAM){
      for(j in 1:N_STATES){
        rn[j+(k-1)*N_STATES] <- paste("Team",k,"State",j)
      }
    }
    cn <- as.vector(matrix(NA, nrow = 1, ncol = TMAX))
    for(k in 1:TMAX){
      cn[k] <- paste("T",k,sep="")
    }
    colnames(t) <- cn
    rownames(t) <- rn
    m_all_forward[[i]] <- t/N_P_PER_TEAM
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
  # Méthode par histogramme, mais peut significative sur événements sparses ((N_CLUSTERS_PROBAS - 1) ème cluster < 0.05)
  # mmsum <- m_all_forward[[1]]
  # for(i in 2:n_matches){
  #   mmsum <- mmsum+m_all_forward[[i]]
  # }
  # mmsum <- mmsum/n_matches
  # h <- hist(mmsum,breaks = seq(0, 1, length.out=prod(dim(mmsum))))
  # co <- h$counts
  # h2 <- hist(cumsum(co), breaks = seq(0, max(cumsum(co)), length.out=N_CLUSTERS_PROBAS+1))
  # co2 <- h2$counts
  # segments <- cumsum(co2)
  # segments_probas <- h$breaks[segments+1]
  # Méthode par histogrammes++
  seuil = 0.001
  mmsum <- m_all_forward[[1]]
  for(i in 2:n_matches){
    mmsum <- mmsum+m_all_forward[[i]]
  }
  mmsum <- mmsum/n_matches
  mmsum[mmsum<seuil] <- 0
  mmsum2 <- mmsum[mmsum!=0]
  h <- hist(mmsum2,breaks = seq(min(mmsum2), 1, length.out=1000))
  co <- h$counts
  h2 <- hist(cumsum(co), breaks = seq(0, max(cumsum(co)), length.out=N_CLUSTERS_PROBAS))
  co2 <- h2$counts
  segments <- cumsum(co2)
  segments_probas <- c(0,h$breaks[segments+1])
  # Méthode linéaire
  # segments_probas <- seq(0,1,length.out=N_CLUSTERS_PROBAS+1)[2:N_CLUSTERS_PROBAS+1]
  
  # Reformatage de mmm pour avoir les segments au lieu des probas
  print("--- Reformatage des probabilites a l'aide des segments")
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
} else {
  print("--- Mise en forme et sommage")
  m_all_forward <- vector("list", N_TEAM)
  for(i in 1:N_TEAM){
    m_all_forward[[i]] <- vector("list", n_matches)
  }
  for(i in 1:n_matches){
    t <- matrix(data = 0, nrow = N_STATES, ncol = TMAX)
    for(j in 1:N_TEAM){
      for(l in 1:TMAX){
        v <- vector("numeric", N_STATES)
        for(k in 1:N_P_PER_TEAM){
          v <- v + m_forward[[j]][[k+(i-1)*N_P_PER_TEAM]][,l]
        }
        t[,l] <- v
      }
      colnames(t) <- colnames(r[[j]]$forward_probs[,,1])
      rownames(t) <- rownames(r[[j]]$forward_probs[,,1])
      m_all_forward[[j]][[i]] <- t
    }
  }
  
  # Reorganisation suivants les canaux (= les etats caches des HMMs inferieurs)
  # m_all_forward est organise suivant les matchs, et pas suivants les canaux
  print("--- Reorganisation des donnees suivant les canaux")
  mmm <- vector("list", N_TEAM)
  mmm2 <- vector("list", N_TEAM)
  for(k in 1:N_TEAM){
    mmm[[k]] <- vector("list", N_STATES)
    mmm2[[k]] <- vector("list", N_STATES)
    for (i in 1:N_STATES){
      mtmp <- matrix(0, n_matches, TMAX)
      rownames(mtmp) <- 1:n_matches
      colnames(mtmp) <- colnames(r[[k]]$forward_probs[,,1])
      for (j in 1:n_matches){
        mtmp[j,] <- m_all_forward[[k]][[j]][i,]
      }
      mmm[[k]][[i]] <- mtmp
    }
  }
  
  # Creation du clustering pour les probas
  # L'idee, c'est de separer les probas (continues) en differents clusters (discrets)
  # afin de pouvoir fournir des classes a emettre par le HMM
  print("--- Definition du clustering des probabilites sur les etats caches")
  
  # Méthode linéaire
  segments_probas <- seq(0,N_P_PER_TEAM,length.out=N_CLUSTERS_PROBAS+1)[2:N_CLUSTERS_PROBAS+1]
  
  # Reformatage de mmm pour avoir les segments au lieu des probas
  print("--- Reformatage des probabilites a l'aide des segments")
  for(i in 1:N_TEAM){
    for(k in 1:N_STATES){
      mtmp <- mmm[[i]][[k]]
      for(j in N_CLUSTERS_PROBAS:1){
        mtmp[mmm[[i]][[k]] <= segments_probas[j]] <- j
      }
      mmm2[[i]][[k]] <- mtmp
    }
  }
  
  print("--- Definition des sequences pour l'apprentissage superieur")
  seqdef.mmm <- vector("list", N_TEAM)
  for(i in 1:N_TEAM){
    seqdef.mmm[[i]] <- vector("list", N_STATES)
    for(j in 1:N_STATES){
      seqdef.mmm[[i]][[j]] <- seqdef(
        data = mmm2[[i]][[j]],
        alphabet = 1:N_CLUSTERS_PROBAS
      )
    }
  }
  
  # Creation du mega hmm
  print("--- Initialisation du HMM superieur")
  tpm <- gen_prob_matrix(N_STATES2, N_STATES2, triangle = TRIANGLE)
  rho <- vector("list", N_STATES)
  for(i in 1:(N_STATES)){
    rho[[i]] <- gen_prob_matrix(N_STATES2, N_CLUSTERS_PROBAS)
  }
  init <- runif(N_STATES2)
  init <- init/sum(init)
  
  megahmms <- vector("list")
  megahmms$hmms <- vector("list", N_TEAM)
  megahmms$fitted.hmms <- vector("list", N_TEAM)
  
  print("--- Apprentissage des HMMs superieur")  
  for(i in 1:N_TEAM){
    megahmms$hmms[[i]] <- build_hmm(
      observations = seqdef.mmm[[i]],
      initial_probs = init,
      emission_probs = rho,
      transition_probs = tpm,
      channel_names = rownames(r[[i]]$forward_probs[,,1])
    )
    print(paste("    --- MegaHMM",i))
    megahmms$fitted.hmms[[i]] <- fit_model(megahmms$hmms[[i]], threads = N_THREADS)
  }
}

# ====================================================-- #
# ============= Debuts de systeme d'analyse ============ #
# ====================================================-- #
# ======================== TODO ======================== #
# ====================================================-- #

# Uniquement si les canaux sont binaires, à améliorer pour le faire pour tout alphabet


# Pour le mega hmm
states <- vector("list", N_STATES2)
for(i in 1:N_STATES2){
  state <- matrix(0,2*N_STATES,10)
  for(j in 1:(2*N_STATES)){
    state[j,] <- fmegahmm$model$emission_probs[[j]][i,]
  }
  rownames(state) <- names(fmegahmm$model$emission_probs)
  states[[i]] <- state
  colnames(states[[i]]) <- 1:10
}

# Part 1
layout(t(matrix(1:30, nrow = 6, ncol = 5)))
for(i in 1:30){myImagePlot(states[[i]])}
layout(1)
# Part 2
layout(t(matrix(1:30, nrow = 6, ncol = 5)))
for(i in 1:30){myImagePlot(states[[i+30]])}
layout(1)
