# Chargement de tout ce qu'il faut
setwd("~/Dropbox/Thèse/R/")
library("rjson")
library("hmm.discnp")
library("seqHMM")
require(graphics)
source("./myImagePlot.R")
source("./gen_prob_matrix.R")
load("./matches_10000.RData")
load("./m_comparatif_loglik.RData")
load("./m_distance_loglik.RData")


json_file <- "../Données/matches1.json"
json_data <- fromJSON(paste(readLines(json_file), collapse=""))
c <- "../Données/champions.json"
c <- fromJSON(paste(readLines(c), collapse=""))
champions <- vector()
len <- 10*length(matches)
                 
# récupération des persos
champions <- vector()
for(i in 1:length(c$data)){
  champions[c$data[[i]]$id] <- c$data[[i]]$name
}
# Récupération des données
data <- vector("list",len)
data_t <- vector("list",len)
pos <- vector("list",len)
winners <- vector()
champions_used <- vector()
for(i in 1:len){
  data[[i]] <- vector()
  pos[[i]]$xs <- vector()
  pos[[i]]$ys <- vector()
}
maxx <- 0
minn <- Inf
for(l in 1:length(matches)){
  fs <- matches[[l]]$timeline$frames
  # maxx <- max(maxx, length(fs))
  # minn <- min(minn, length(fs))
  # for(i in 1:10){
  #   p <- matches[[l]]$participants[[i]]
  #   champions_used[p$participantId+10*(l-1)] <- p$championId
  #   winners[p$participantId+10*(l-1)] <- p$stats$winner
  # }
  for(i in 1:length(fs)){
    f <- fs[[i]]
    es <- f$events
  #   for(j in 1:length(es)){
  #     e <- es[[j]]
  #     if(!is.null(e$participantId)){
  #       if(e$participantId != 0){
  #         data[[e$participantId+10*(l-1)]] <- append(data[[e$participantId+10*(l-1)]], e$eventType)
  #         data_t[[e$participantId+10*(l-1)]] <- append(data_t[[e$participantId+10*(l-1)]], e$timestamp)
  #       }
  #     }
  #     else if(!is.null(e$creatorId)){
  #       if(e$creatorId != 0) {
  #         data[[e$creatorId+10*(l-1)]] <- append(data[[e$creatorId+10*(l-1)]], e$eventType)
  #         data_t[[e$creatorId+10*(l-1)]] <- append(data_t[[e$creatorId+10*(l-1)]], e$timestamp)
  #       }
  #     }
  #     else if(!is.null(e$killerId)){
  #       if(e$killerId != 0){
  #         data[[e$killerId+10*(l-1)]] <- append(data[[e$killerId+10*(l-1)]], e$eventType)
  #         data_t[[e$killerId+10*(l-1)]] <- append(data_t[[e$killerId+10*(l-1)]], e$timestamp)
  #       }
  #       if(!is.null(e$assistingParticipantIds)){
  #         api <- e$assistingParticipantIds
  #         for(k in 1:length(api)){
  #           data[[api[k]+10*(l-1)]] <- append(data[[api[k]+10*(l-1)]], paste(e$eventType,"_ASSIST",sep=""))
  #           data_t[[api[k]+10*(l-1)]] <- append(data_t[[api[k]+10*(l-1)]], e$timestamp)
  #         }
  #       }
  #       if(!is.null(e$victimId)){
  #         data[[e$victimId+10*(l-1)]] <- append(data[[e$victimId+10*(l-1)]], "IS_KILLED")
  #         data_t[[e$victimId+10*(l-1)]] <- append(data_t[[e$victimId+10*(l-1)]], e$timestamp)
  #       }
  #     }
  #   }
    for(j in 1:length(f$participantFrames)){
      pf <- f$participantFrames[[j]]
      pos[[pf$participantId+10*(l-1)]]$xs <- append(pos[[pf$participantId+10*(l-1)]]$xs ,pf$position$x)
      pos[[pf$participantId+10*(l-1)]]$ys <- append(pos[[pf$participantId+10*(l-1)]]$ys ,pf$position$y)
    }
  }
}

xxs <- vector("list", len)
for(i in 1:len){
  xxs[[i]] <- pos[[i]]$xs
}

yys <- vector("list", len)
for(i in 1:len){
  yys[[i]] <- pos[[i]]$ys
}

# Définition du vocabulaire total (moins efficace mais plus court)
vocab <- vector()
for(i in 1:length(data)){
  vocab <- union(vocab, data[[i]])
}
rm(e, es, f, fs, i, j, json_file)

#========= HMM de hmm.discnp, peut charger ./hmm_1000.RData pour éviter apprentissage
hmm <- hmm(data, K = 50,
           rand.start=list(Rho=TRUE,tpm=TRUE),
           verbose=TRUE)

# Viterbi sur chaque séquence d'événements
vit <- vector("list",len)
for(i in 1:(len)){
  print(i)
  vit[[i]] <- viterbi(y = data[[i]],
                      object = hmm)
}

#=========== Première analyze via K-means
### avec cette analyse, on perd la dimension temporelle de la séquence d'actions
### extraction des coordonnées sur les 50 noeuds
vit2 <- vector("list",len)
for(i in 1:(len)){
  vit2[[i]] <- table(vit[[i]])
}
### création de la matrice des coordonnées
m_data <- matrix(ncol = 50, nrow = length(data), data = 0)
for(i in 1:length(vit2)){
  m_data[i,as.integer(names(vit2[[i]]))] <- as.vector(vit2[[i]])
}

### normalisation
##### on normalise sur les joueurs afin de garder la prépondérance de chaque noeud
m_data <- apply(m_data,1,function(line){
  return((line-mean(line))/var(line))
})
### k_means
km <- kmeans(m_data, 5)

### récupération des persos dans chaque cluster
clusters <- vector("list",5)
for(i in 1:5){
  clusters[[i]] <- champions_used[which(km$cluster==i)]
}

#========== Deuxième analyse via sequence clustering
hmms <- vector("list", len)
### apprentissage d'un hmm par séquence
for(i in 1:len){
  print(i)
  K <- 11
  Rho <- gen_prob_matrix(length(vocab), K)
  rownames(Rho) <- vocab
  tpm <- gen_prob_matrix(K, K)
  hmms[[i]] <- hmm(data[[i]],
                   K = K,
                   par0 = list(Rho = Rho, tpm = tpm))
}
### matrice des comparatifs : L_ij = LogLik(sequence_i | hmm_j)
m_comparatif <- matrix(data=0, nrow=len, ncol=len)
for(i in 1:len){
  print(i)
  for(j in 1:len){
    m_comparatif[i,j] <- logLikHmm(data[[i]],hmms[[j]])
  }
}
### matrice des distances : D_ij = 1/2*(L_ij + L_ji)
m_distance <- matrix(data=0, nrow = len, ncol = len)
for(i in 1:len){
  print(i)
  for(j in 1:len){
    if(i==j){
      m_distance[i,j] <- 0
    } else {
      m_distance[i,j] <- (m_comparatif[i,j]+m_comparatif[j,i])/2
      m_distance[j,i] <- (m_comparatif[i,j]+m_comparatif[j,i])/2
    }
  }
}
m_distance <- as.dist(m_distance)

### clustering hierarchique
hc <- hclust(m_distance)
members <- cutree(hc, k_clusters)

### apprentissage d'un hmm par cluster
k_clusters = 10
k_hidden_states = 11
hmms_clusters = vector("list", k_clusters)
winners_clusters = vector("list", k_clusters)
for(i in 1:k_clusters){
  data_cluster <- data[which(members==i)]
  winners_clusters[[i]] <- winners[which(members==i)]
  hmms_clusters[[i]] <- hmm(data_cluster,
                            K = k_hidden_states,
                            rand.start=list(Rho=TRUE,tpm=TRUE),
                            verbose = TRUE)
}
### pourcentage de winner par cluster
for(i in 1:k_clusters){
  w <- winners_clusters[[i]]
  print(paste("cluster", i, "with", length(w), "individuals", sep=" "))
  print(paste("--------", sum(w)/length(w)))
}

#=========== Visualisation
hmm <- hmms_clusters[[4]]
myImagePlot(hmm$Rho)
myImagePlot(hmm$tpm)
for(i in 1:5){
  print(paste("======= Cluster ", i, sep = ""))
  print(sort(table(clusters[[i]])))
}
sort(table(champions_used[which(members==1)]))