setwd("~/Dropbox/Thèse/R/") # Xubuntu
# setwd("../Dropbox/Thèse/R/") # Windows
library("seqHMM")
library("rjson")
source("./gen_prob_matrix.R")
source("./myImagePlot.R")
source("./transfer_data.R")
load("./package_data_lol_NoTMAX.RData")
load("./results_lol_1megahmm_app-test_40HMMsup.RData")
load("./results_lol_1megahmm_app-test.RData")

len <- 9000
need_seq = T
N_STATES <-20
N_CLUSTERS_PROBAS <- 10
N_TEAM <- 2
N_P_PER_TEAM <- 5
TMAX <- 60 # <=  la duree minimum d'une partie pour le moment. <- l_max si pas de limitation
TRIANGLE <- F
N_THREADS <- 4
MODE_PER_TEAM <- F
SEUIL_REPRESENTATIVITE <- 0.01
n_matches <- 900
for(i in 13:1){
  matrice.global.test[[i]] <- NULL
  alphabets[[i]] <- NULL
}
vocab <- vocab[14:15]

means <- vector("numeric", length(15:60))
nNan <- vector("numeric", length(15:60))

seqdef.test <- vector("list")
seqdef.test$teams <- vector("list", N_TEAM)
d <- dim(matrice.global.test[[1]])
n_matches_test <- d[1]/N_P_PER_TEAM/N_TEAM
for(i in 1:N_TEAM){
  seqdef.test$teams[[i]]$matrices <- vector("list", length(matrice.global.test))
  seqdef.test$teams[[i]]$seq <- vector("list", length(matrice.global.test))
  for(j in 1:length(matrice.global.test)){
    seqdef.test$teams[[i]]$matrices[[j]] <- matrix(NA, nrow = d[1]/2, ncol = d[2])
    for(k in 1:(d[1]/10)){
      seqdef.test$teams[[i]]$matrices[[j]][1:N_P_PER_TEAM+(k-1)*N_P_PER_TEAM,] <- matrice.global.test[[j]][1:N_P_PER_TEAM+(k-1)*N_P_PER_TEAM*N_TEAM+(i-1)*N_P_PER_TEAM,]
    }
  }
}
for(i in 1:length(matrice.global.test)){
  for(j in 1:N_TEAM){
    seqdef.test$teams[[j]]$seq[[i]] <- seqdef(
      data = seqdef.test$teams[[j]]$matrice[[i]],
      alphabet = alphabets[[i]])      
  }
}

for(yolo in 15:60){
  N_STATES2 <- yolo
  print(paste("Test pour E_2 =", yolo))

  
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
    
    hmms.test <- vector("list",N_TEAM)
    for(i in 1:N_TEAM){
      # Uniquement pour les positions
      hmms.test[[i]] <- transfer_data(seqdef.test$teams[[i]]$seq, hmm.per.team$fitted.hmms[[i]]$model)
    }
    
    # RECOPIE MISE EN FORME DES DONNEES FORWARD
    print("--- Recuperation des forwards probabilities")
    r <- vector("list", N_TEAM)
    for(i in 1:N_TEAM){
      r[[i]] <- forward_backward(hmms.test[[i]], forward_only = T, threads = N_THREADS)
    }
    # m_forward[[équipe]][[séquence d'un joueur de cette team]]
    m_forward <- vector("list", N_TEAM)
    for(i in 1:N_TEAM){
      m_forward[[i]] <- vector("list", n_matches_test*N_P_PER_TEAM)
    }
    for(i in 1:n_matches_test){
      for(j in 1:N_P_PER_TEAM){
        for(k in 1:N_TEAM){
          m_forward[[k]][[(i-1)*N_P_PER_TEAM+j]] <- r[[k]]$forward_probs[,,(i-1)*N_P_PER_TEAM+j]
        }
      }
    }
    print("--- Mise en forme et sommage")
    m_all_forward <- vector("list", n_matches_test)
    for(i in 1:n_matches_test){
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
    
    print("--- Reorganisation des donnees suivant les canaux")
    mmm <- vector("list", N_TEAM*N_STATES)
    mmm2 <- vector("list", N_TEAM*N_STATES)
    for (i in 1:(N_TEAM*N_STATES)){
      mtmp <- matrix(0, n_matches_test, TMAX)
      rownames(mtmp) <- 1:n_matches_test
      colnames(mtmp) <- colnames(r$forward_probs[,,1])
      for (j in 1:n_matches_test){
        mtmp[j,] <- m_all_forward[[j]][i,]
      }
      mmm[[i]] <- mtmp
    }
    
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
    
    # Hmm sup unique
    megahmm.test <- transfer_data(seqdef.mmm, fmegahmm$model)
    a <- logLik(megahmm.test, partial = T)
    a <- as.vector(a)
    nNan[yolo-14] <- sum(is.na(a))
    a <- a[!is.na(a)]
    means[yolo-14] <- mean(a)
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
    # Méthode par histogramme, mais peut significative sur événements sparses ((N_CLUSTERS_PROBAS - 1) ème cluster < 0.05)
    seuil = 0.001
    d <- dim(m_all_forward[[1]][[1]])
    mmsum <- matrix(0, nrow = d[1], ncol = d[2])
    for(j in 1:N_TEAM){
      for(i in 1:n_matches){
        mmsum <- mmsum+m_all_forward[[j]][[i]]
      }    
    }
    
    mmsum <- mmsum/n_matches/N_P_PER_TEAM
    mmsum[mmsum<seuil] <- 0
    mmsum2 <- mmsum[mmsum!=0]
    h <- hist(mmsum2,breaks = seq(min(mmsum2), 1, length.out=1000))
    co <- h$counts
    h2 <- hist(cumsum(co), breaks = seq(0, max(cumsum(co)), length.out=N_CLUSTERS_PROBAS))
    co2 <- h2$counts
    segments <- cumsum(co2)
    segments_probas <- c(0,h$breaks[segments+1])
    
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
}

a <- forward_backward(megahmm.test, log_space = T )
for(i in 1:100){
  s <- a$forward_probs[,,i]
  s[s==-Inf] <- NA
}




