require(dgof)

elements <- 1:20


data_dict <- list(
  all = list(
    observed_counts = c(78572726, 1110863, 111483, 19885, 6792, 2776, 1370, 679, 379, 246, 148, 86, 58, 55, 28, 19, 14, 11, 8, 5),
    expected_cumulative_proportions = c(0, 0.9373018416, 0.9959973777, 0.9996804727, 0.9999606849, 0.9999943519, 0.9999965138, 0.9999996202, 0.9999999771, 0.9999999771, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
  ),
  excitatory = list(
    observed_counts = c(60199273, 757277, 51891, 5541, 1294, 334, 122, 38, 28, 9, 4, 4, 0, 1, 0, 0, 0, 0, 1, 0),
    expected_cumulative_proportions = c(0, 0.9412661102, 0.9961815693, 0.9999066798, 0.9999782302, 0.9999989234, 0.9999996462, 0.9999996918, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
  ),
  inhibitory = list(
    observed_counts = c(18373453, 353586, 59592, 14344, 5498, 2442, 1248, 641, 351, 237, 144, 82, 58, 54, 28, 19, 14, 11, 7, 5),
    expected_cumulative_proportions = c(0, 0.9252772537, 0.9953773483, 0.999000039, 0.9999042753, 0.9999801006, 0.9999865658, 0.9999994211, 0.9999998971, 0.9999998971, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
  )
)

for (dtype in names(data_dict)) {
    cat('For datatype', dtype, 'axons')
    expected_cum_prop <-  data_dict[[dtype]]$expected_cumulative_proportions
    observed_counts <-  data_dict[[dtype]]$observed_counts

    observed_instances <- unlist(lapply(1:length(elements), function(i) rep(elements[i], times = observed_counts[i])))

    expected_cdf <- stepfun(elements, expected_cum_prop)

    ks_results <- ks.test(observed_instances, expected_cdf, alternative = c("two.sided"), simulate.p.value = TRUE, B=50)
    print(ks_results)

}





list('all')
#cvm_results <- cvm.test(observed_instances, expected_cdf, type = c("W2"), simulate.p.value = FALSE, B=200)

