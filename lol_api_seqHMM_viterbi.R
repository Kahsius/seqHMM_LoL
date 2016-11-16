setwd("~/Dropbox/Thèse/seqHMM_LoL/") # Xubuntu
# setwd("../Dropbox/Thèse/R/") # Windows
library("seqHMM")
library("rjson")
source("./gen_prob_matrix.R")
source("./myImagePlot.R")
source("./transfer_data.R")

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
if(!exists("matrice.global")){
  load("./package_data_lol_NoTMAX.RData")
  load("./results_lol_1megahmm_app-test_40HMMsup.RData")
  load("./results_lol_1megahmm_app-test.RData")
}

#Test débile hors système, on étudie uniquement les positions
if(T){
  if(length(matrice.global) != 2){
    for(i in 13:1){
      matrice.global[[i]] <- NULL
    }
  }
  for(i in 13:1){
    matrice.global.test[[i]] <- NULL
    alphabets[[i]] <- NULL
  }
  vocab <- vocab[14:15]  
}


len <- dim(matrice.global[[1]])[1]
need_seq = !exists("seq.each.team")
# Extraction de la longueur maximale
l_max <- max(apply(matrice.global[[1]], 1, function(row){
  return(sum(!is.na(row))) #
}))

N_STATES <- 40
N_STATES2 <- 60
# N_CLUSTERS_PROBAS <- 10
N_TEAM <- 2
N_P_PER_TEAM <- 5
TMAX <- l_max # <=  la duree minimum d'une partie pour le moment. <- l_max si pas de limitation
TRIANGLE <- F
N_THREADS <- 4
MODE_PER_TEAM <- T
SEUIL_REPRESENTATIVITE <- 0.01
n_matches <- len/N_P_PER_TEAM/N_TEAM

# Sort les événements non pertinents
# for(i in length(vocab):1){
#   if(sum(matrice.global[[i]])/prod(dim(matrice.global[[i]])) < SEUIL_REPRESENTATIVITE) {
#     matrice.global[[i]] <- NULL
#     vocab <- vocab[-i]
#     alphabets[[i]] <- NULL
#   }
# }

print("Definition des sequences pour l'apprentissage du futur mixture modele")
# Changement du format de donnees en sequences pour l'ensemble des donnees
if(need_seq){
  
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
} else {
  print("-- Récupération des données par Viterbi")
  v <- vector("list", N_TEAM) # Viterbi's paths
  m <- vector("list", N_TEAM) # Matrices of index along Viterbi's paths per team
  for(i in 1:N_TEAM){
    print(paste("   -- Viterbi équipe :",i))
    # Objet qui contient tous les chemins de viterbi
    v[[i]] <- as.matrix(hidden_paths(hmm.per.team$fitted.hmms[[i]]$model))
    max_len <- max(apply(v[[i]], MARGIN = 1, length)) # on récupère la longueur de la plus grande chaîne
    n_seq <- dim(v[[i]])[1]
    m[[i]]$data <- matrix(nrow = n_seq, ncol = max_len) # la matrice de données
    m[[i]]$per_state <- vector("list", N_STATES) # les données par état caché
    l_matches <- vector("numeric", n_seq/N_P_PER_TEAM)
    
    for(j in 1:N_STATES){
      m[[i]]$per_state[[j]] <- matrix(nrow = n_seq/N_P_PER_TEAM, ncol = max_len)
    }
    for(j in 1:n_seq){
      # Là on essaye d'outrepasser seqHMM en lui demandant de prendre uniquement les prévisions
      # pour la durée de la partie
      obs_seq <- as.matrix(hmm.per.team$fitted.hmms[[i]]$model$observations[[1]])[j,]
      l_seq <- length(obs_seq[obs_seq != "%"])
      l_matches[j%/%N_P_PER_TEAM+1] <- l_seq
      tmp <- v[[i]][j,]
      tmp <- sapply(tmp, function(e){return(strsplit(e, split=" "))})
      for(k in 1:l_seq){
        # On injecte les données dans les matrices correspondantes
        m[[i]]$data[j,k] <- as.numeric(tmp[[k]][2])
      }
    }
    for(j in 1:(n_seq/N_P_PER_TEAM)){
      seq_team <- m[[i]]$data[1:N_P_PER_TEAM+(j-1)*N_P_PER_TEAM,]
      for(k in 1:N_STATES){
        for(l in 1:l_matches[j]){
          m[[i]]$per_state[[k]][j,l] <- sum(seq_team[,l] == k)
        }
      }
    }
  }
  
  # Les index des joueurs de l'équipe 1 dans l'ensemble d'apprentissage
  app.index.t1 <- as.vector(sapply(which(!(1:1000 %in% index)), function(n){
    return(1:5+(n-1)*10)
  }))
  # Les victoires ou non des joueur de t1
  app.rank.t1 <- winners[app.index.t1]
  # Les gagnants dans t1
  app.winners.t1 <- which(app.rank.t1)
  # Les perdants dans t2
  app.losers.t1 <- which(!app.rank.t1)
  # Les matrices des données des gagnants et des perdants
  m.winners.t1 <- m[[1]]$data[app.winners.t1,]
  m.losers.t1 <- m[[1]]$data[app.losers.t1,]
  t.winners.t1 <- table(m.winners.t1)
  t.losers.t1 <- table(m.losers.t1)
  w <- t.winners.t1/sum(t.winners.t1)
  l <- t.losers.t1/sum(t.losers.t1)
  diff <- w-l
  
  
  
  print("--- Definition des sequences pour l'apprentissage superieur")
  seqdef.m <- vector("list", N_TEAM)
  for(i in 1:N_TEAM){
    seqdef.m[[i]]$seq <- vector("list", N_STATES)
    for(j in 1:N_STATES){
      seqdef.m[[i]]$seq[[j]] <- seqdef(
        data = m[[i]]$per_state[[j]],
        alphabet = 0:N_P_PER_TEAM
      )
    }
  }
  
  # Creation du mega hmm
  print("--- Initialisation du HMM superieur")
  tpm <- gen_prob_matrix(N_STATES2, N_STATES2, triangle = TRIANGLE)
  rho <- vector("list", N_STATES)
  for(i in 1:(N_STATES)){
    rho[[i]] <- gen_prob_matrix(N_STATES2, N_P_PER_TEAM+1)
  }
  init <- runif(N_STATES2)
  init <- init/sum(init)
  
  megahmms <- vector("list")
  megahmms$hmms <- vector("list", N_TEAM)
  megahmms$fitted.hmms <- vector("list", N_TEAM)
  
  print("--- Apprentissage des HMMs superieur")  
  for(i in 1:N_TEAM){
    megahmms$hmms[[i]] <- build_hmm(
      observations = seqdef.m[[i]]$seq,
      initial_probs = init,
      emission_probs = rho,
      transition_probs = tpm,
      channel_names = sapply(1:N_STATES, function(e){return(paste("State ",e))})
    )
    print(paste("    --- MegaHMM",i))
    megahmms$fitted.hmms[[i]] <- fit_model(megahmms$hmms[[i]], threads = N_THREADS)
  }
  
  # ===============================================================================================
  # Définition des données de test
  
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
  for(i in 13:1){
    for(j in 1:N_TEAM)
    seqdef.test$teams[[j]]$seq[[i]] <- NULL
  }
  hmms.test <- vector("list",N_TEAM)
  for(i in 1:N_TEAM){
    # Uniquement pour les positions
    hmms.test[[i]] <- transfer_data(seqdef.test$teams[[i]]$seq, hmm.per.team$fitted.hmms[[i]]$model)
  }
  
  print("-- Récupération des données par Viterbi")
  v <- vector("list", N_TEAM) # Viterbi's paths
  m <- vector("list", N_TEAM) # Matrices of index along Viterbi's paths per team
  for(i in 1:N_TEAM){
    print(paste("   -- Viterbi équipe :",i))
    # Objet qui contient tous les chemins de viterbi
    v[[i]] <- as.matrix(hidden_paths(hmms.test[[i]]))
    max_len <- max(apply(v[[i]], MARGIN = 1, length)) # on récupère la longueur de la plus grande chaîne
    n_seq <- dim(v[[i]])[1]
    m[[i]]$data <- matrix(nrow = n_seq, ncol = max_len) # la matrice de données
    m[[i]]$per_state <- vector("list", N_STATES) # les données par état caché
    l_matches <- vector("numeric", n_seq/N_P_PER_TEAM)
    
    for(j in 1:N_STATES){
      m[[i]]$per_state[[j]] <- matrix(nrow = n_seq/N_P_PER_TEAM, ncol = max_len)
    }
    for(j in 1:n_seq){
      # Là on essaye d'outrepasser seqHMM en lui demandant de prendre uniquement les prévisions
      # pour la durée de la partie
      obs_seq <- as.matrix(hmms.test[[i]]$observations[[1]])[j,]
      l_seq <- length(obs_seq[obs_seq != "%"])
      l_matches[j%/%N_P_PER_TEAM+1] <- l_seq
      tmp <- v[[i]][j,]
      tmp <- sapply(tmp, function(e){return(strsplit(e, split=" "))})
      for(k in 1:l_seq){
        # On injecte les données dans les matrices correspondantes
        m[[i]]$data[j,k] <- as.numeric(tmp[[k]][2])
      }
    }
    for(j in 1:(n_seq/N_P_PER_TEAM)){
      seq_team <- m[[i]]$data[1:N_P_PER_TEAM+(j-1)*N_P_PER_TEAM,]
      for(k in 1:N_STATES){
        for(l in 1:l_matches[j]){
          m[[i]]$per_state[[k]][j,l] <- sum(seq_team[,l] == k)
        }
      }
    }
  }
  
  # print("--- Definition des sequences pour l'apprentissage superieur")
  # seqdef.m.test <- vector("list", N_TEAM)
  # for(i in 1:N_TEAM){
  #   seqdef.m.test[[i]]$seq <- vector("list", N_STATES)
  #   for(j in 1:N_STATES){
  #     seqdef.m.test[[i]]$seq[[j]] <- seqdef(
  #       data = m[[i]]$per_state[[j]],
  #       alphabet = 0:N_P_PER_TEAM
  #     )
  #   }
  # }
  # 
  # megahmm.test <- transfer_data(seqdef.m.test[[ihmm]]$seq, megahmms$fitted.hmms[[1]]$model)
}

# ====================================================-- #
# ============= Debuts de systeme d'analyse ============ #
# ====================================================-- #
# ======================== TODO ======================== #
# ====================================================-- #

if(MODE_PER_TEAM){
  ihmm <- 1
  
  # Pour le mega hmm
  fmegahmm <- megahmms$fitted.hmms[[ihmm]]
  states <- vector("list", N_STATES2)
  for(i in 1:N_STATES2){
    state <- matrix(0,N_STATES,N_P_PER_TEAM+1)
    for(j in 1:N_STATES){
      state[j,] <- fmegahmm$model$emission_probs[[j]][i,]
    }
    rownames(state) <- names(fmegahmm$model$emission_probs)
    states[[i]] <- state
    colnames(states[[i]]) <- 0:N_P_PER_TEAM
  }
  
  hmm <- hmm.per.team$fitted.hmms[[ihmm]]
  images <- vector("list", N_STATES2)
  images_inf1 <- vector("list", N_STATES)
  for(i in 1:N_STATES2){
    image <- matrix(0, nrow = 15, ncol = 15)
    for(j in 1:N_STATES){
      coef <- as.numeric(t(states[[i]][j,])%*%0:N_P_PER_TEAM)
      xx <- hmm$model$emission_probs$x[j,]
      yy <- hmm$model$emission_probs$y[j,]
      mx <- replicate(15,xx)
      my <- t(replicate(15,yy))
      images_inf1[[j]] <- coef*mx*my
      image <- image + coef*mx*my
    }
    images[[i]] <- image
  }
  
  megahmm.test <- transfer_data(seqdef.mmm, fmegahmm$model)
  a <- logLik(megahmm.test, partial = T)
} else {
  # Pour le mega hmm unique
  model <- megahmm.test
  states <- vector("list", N_STATES2)
  for(i in 1:N_STATES2){
    state <- matrix(0,2*N_STATES,N_CLUSTERS_PROBAS)
    for(j in 1:(2*N_STATES)){
      state[j,] <- model$emission_probs[[j]][i,]
    }
    rownames(state) <- names(model$emission_probs)
    states[[i]] <- state
    colnames(states[[i]]) <- 1:N_CLUSTERS_PROBAS
  }
  
  hmm1 <- hmm.per.team$fitted.hmms[[1]]
  images1 <- vector("list", N_STATES2)
  for(i in 1:N_STATES2){
    image <- matrix(0, nrow = 15, ncol = 15)
    for(j in 1:N_STATES){
      coef <- sum(states[[i]][j,]*segments_probas)
      xx <- hmm1$model$emission_probs$x[j,]
      yy <- hmm1$model$emission_probs$y[j,]
      mx <- replicate(15,xx)
      my <- t(replicate(15,yy))
      image <- image + coef*mx*my
    }
    images1[[i]] <- image
  }
  
  hmm2 <- hmm.per.team$fitted.hmms[[2]]
  images2 <- vector("list", N_STATES2)
  for(i in 1:N_STATES2){
    image <- matrix(0, nrow = 15, ncol = 15)
    for(j in 1:N_STATES){
      coef <- sum(states[[i]][j+N_STATES,]*segments_probas)
      xx <- hmm2$model$emission_probs$x[j,]
      yy <- hmm2$model$emission_probs$y[j,]
      mx <- replicate(15,xx)
      my <- t(replicate(15,yy))
      image <- image + coef*mx*my
    }
    images2[[i]] <- image
  }
  
  images <- vector("list", N_STATES2)
  for(i in 1:N_STATES2){
    images[[i]] <- images1[[i]] + images2[[i]]
  }
  
  # images_inf1 <- vector("list", N_STATES)
  # hmm1 <- hmm.per.team$fitted.hmms[[1]]
  # for(i in 1:N_STATES){
  #   xx <- hmm1$model$emission_probs$x[i,]
  #   yy <- hmm1$model$emission_probs$y[i,]
  #   mx <- replicate(15,xx)
  #   my <- t(replicate(15,yy))
  #   images_inf1[[i]] <-mx*my
  # }
  # 
  # images_inf2 <- vector("list", N_STATES)
  # hmm2 <- hmm.per.team$fitted.hmms[[2]]
  # for(i in 1:N_STATES){
  #   xx <- hmm2$model$emission_probs$x[i,]
  #   yy <- hmm2$model$emission_probs$y[i,]
  #   mx <- replicate(15,xx)
  #   my <- t(replicate(15,yy))
  #   images_inf2[[i]] <-mx*my
  # }
}

# TRUKPAPROPRES #
if(FALSE){
  # ================================================================================================
  
  # Mega HMM
  layout(t(matrix(1:N_STATES2, nrow = 10, ncol = ceiling(N_STATES2/10))))
  for(i in 1:N_STATES2){myImagePlot(images[[i]])}
  layout(1)
  
  # HMM1
  layout(t(matrix(1:20, nrow = 4, ncol = 5)))
  for(i in 1:20){myImagePlot(images_inf1[[i]])}
  layout(1)
  
  # HMM2
  layout(t(matrix(1:20, nrow = 4, ncol = 5)))
  for(i in 1:20){myImagePlot(images_inf2[[i]])}
  layout(1)
  
  # ================================================================================================
  
  i <- 2
  model <- megahmms.test
  
  #Jouli animation
  viterbi <- as.matrix(hidden_paths(model))
  v1 <- viterbi[i,]
  for(i in 1:length(v1)){
    index <- as.numeric(strsplit(v1[i], split=" ")[[1]][2])
    myImagePlot(images[[index]])
    Sys.sleep(.3)
  }
  
  # Enregistrement dans un fichier
  saveGIF({
    i <- 3
    viterbi <- as.matrix(hidden_paths(fmegahmm$model))
    v <- viterbi[i,]
    for(i in 1:length(v)){
      index <- as.numeric(strsplit(v[i], split=" ")[[1]][2])
      myImagePlot(images[[index]])
    }
  })

  # ================================================================================================
  index_match <- 65;
  
  a <- matrice.global.test[[1]][1:10+(index_match-1)*10,]
  b <- matrice.global.test[[2]][1:10+(index_match-1)*10,]
  t <- sum(!is.na(a[1,]))
  run <- vector("list", t)
  for(i in 1:t){run[[i]] <- matrix(0, nrow=15, ncol = 15)}
  for(i in 1:t){
    for(j in 1:5){
      run[[i]][a[j,i]+1,b[j,i]+1] <- run[[i]][a[j,i]+1,b[j,i]+1] + 1
    }
    for(j in 6:10){
      run[[i]][a[j,i]+1,b[j,i]+1] <- run[[i]][a[j,i]+1,b[j,i]+1] - 1
    }
  }
  for(i in 1:length(run)){
    myImagePlot(run[[i]])
    Sys.sleep(1)
  }
}













