def preserving_group_loss(prev_segments_loss, concatenation_loss, factor = 1):
    print("Grouping criterion", prev_segments_loss, concatenation_loss)
    return (
        concatenation_loss < factor * prev_segments_loss
    )