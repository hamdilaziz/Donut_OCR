import re
import editdistance

def keep_all_but_tokens(str, tokens):
    """
    Remove all layout tokens from string
    """
    return re.sub('([' + tokens + '])', '', str)

def format_string_for_cer(str, layout_tokens):
    """
    Format string for CER computation: remove layout tokens and extra spaces
    """
    if layout_tokens is not None:
        str = keep_all_but_tokens(str, layout_tokens)  # remove layout tokens from metric
    str = re.sub('([\n])+', "\n", str)  # remove consecutive line breaks
    str = re.sub('([ ])+', " ", str).strip()  # remove consecutive spaces
    return str

def edit_cer_from_string(gt, pred, layout_tokens=None):
    """
    Format and compute edit distance between two strings at character level
    """
    gt = format_string_for_cer(gt, layout_tokens)
    pred = format_string_for_cer(pred, layout_tokens)
    return editdistance.eval(gt, pred)




########################################################################
def format_string_for_wer(str, layout_tokens):
    """
    Format string for WER computation: remove layout tokens, treat punctuation as word, replace line break by space
    """
    str = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', str)  # punctuation processed as word
    if layout_tokens is not None:
        str = keep_all_but_tokens(str, layout_tokens)  # remove layout tokens from metric
    str = re.sub('([ \n])+', " ", str).strip()  # keep only one space character
    return str.split(" ")

def edit_wer_from_formatted_split_text(gt, pred):
    """
    Compute edit distance at word level from formatted string as list
    """
    return editdistance.eval(gt, pred)

def edit_wer_from_string(gt, pred, layout_tokens=None):
    """
    Format and compute edit distance between two strings at word level
    """
    split_gt = format_string_for_wer(gt, layout_tokens)
    split_pred = format_string_for_wer(pred, layout_tokens)
    return edit_wer_from_formatted_split_text(split_gt, split_pred)


