def get_decision(pred_state, pred_int, stress, energy, timeofday):
    """Decision logic: What + When"""
    if pred_state in ['calm', 'neutral'] and pred_int <= 2:
        return 'deep_work', 'now'
    elif pred_state == 'overwhelmed' or stress > 3:
        return 'box_breathing', 'now'
    elif pred_state == 'restless' or energy < 2:
        if timeofday in ['night', 'evening']:
            return 'rest', 'tonight'
        else:
            return 'movement', 'within_15_min'
    elif pred_int >= 4:
        return 'journaling', 'later_today'
    else:
        return 'grounding', 'now'
