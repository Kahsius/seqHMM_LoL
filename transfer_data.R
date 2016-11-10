transfer_data <- function(data, model){
  # Data doit être au format seqdef
  new <- build_hmm(
    observations = data,
    initial_probs = model$initial_probs,
    emission_probs = model$emission_probs,
    transition_probs = model$transition_probs,
    channel_names = model$channel_names,
    state_names = model$state_names)
  return(new)
}