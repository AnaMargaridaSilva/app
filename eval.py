predictions = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**{k: batch[k] if k in batch else None for k in ["input_ids", "attention_mask", "token_type_ids", "signal_bias_mask"]})
                # ce_predictions = outputs["ce_logits"].argmax(dim=-1).tolist()
                # sig_predictions = outputs["sig_logits"].argmax(dim=-1).tolist()
                start_cause_predictions = outputs["start_arg0_logits"]
                end_cause_predictions = outputs["end_arg0_logits"]

                start_effect_predictions = outputs["start_arg1_logits"]
                end_effect_predictions = outputs["end_arg1_logits"]

                start_signal_predictions = outputs["start_sig_logits"]
                end_signal_predictions = outputs["end_sig_logits"]

                for i in range(len(batch["input_ids"])):
                    word_ids = batch["word_ids"][i]
                    space_splitted_tokens = batch["text"][i].split(" ")

                    if args.postprocessing_position_selector:
                        if not args.beam_search:
                            start_cause, end_cause, start_effect, end_effect = model.position_selector(
                                start_cause_logits=start_cause_predictions[i],
                                end_cause_logits=end_cause_predictions[i],
                                start_effect_logits=start_effect_predictions[i],
                                end_effect_logits=end_effect_predictions[i],
                                attention_mask=batch["attention_mask"][i],
                                word_ids=word_ids,
                            )
                        else:
                            indices1, indices2, score1, score2, topk_scores = model.beam_search_position_selector(
                                start_cause_logits=start_cause_predictions[i],
                                end_cause_logits=end_cause_predictions[i],
                                start_effect_logits=start_effect_predictions[i],
                                end_effect_logits=end_effect_predictions[i],
                                attention_mask=batch["attention_mask"][i],
                                word_ids=word_ids,     
                                topk=args.topk,
                            )
                    else:
                        start_cause_predictions[i] -= (1 - batch["attention_mask"][i]) * 1e4
                        end_cause_predictions[i] -= (1 - batch["attention_mask"][i]) * 1e4
                        start_effect_predictions[i] -= (1 - batch["attention_mask"][i]) * 1e4
                        end_effect_predictions[i] -= (1 - batch["attention_mask"][i]) * 1e4
                    
                        start_cause_predictions[i][0] = -1e4
                        end_cause_predictions[i][0] = -1e4
                        start_effect_predictions[i][0] = -1e4
                        end_effect_predictions[i][0] = -1e4

                        start_cause_predictions[i][len(word_ids) - 1] = -1e4
                        end_cause_predictions[i][len(word_ids) - 1] = -1e4
                        start_effect_predictions[i][len(word_ids) - 1] = -1e4
                        end_effect_predictions[i][len(word_ids) - 1] = -1e4

                        start_cause = start_cause_predictions[i].argmax().item()
                        end_cause = end_cause_predictions[i].argmax().item()
                        start_effect = start_effect_predictions[i].argmax().item()
                        end_effect = end_effect_predictions[i].argmax().item()
                    
                    has_signal = 1
                    if args.signal_classification:
                        if not args.pretrained_signal_detector:
                            has_signal = outputs["signal_classification_logits"][i].argmax().item()
                        else:
                            has_signal = signal_detector.predict(text=batch["text"][i])

                    if has_signal:
                        start_signal_predictions[i] -= (1 - batch["attention_mask"][i]) * 1e4
                        end_signal_predictions[i] -= (1 - batch["attention_mask"][i]) * 1e4

                        start_signal_predictions[i][0] = -1e4
                        end_signal_predictions[i][0] = -1e4

                        start_signal_predictions[i][len(word_ids) - 1] = -1e4
                        end_signal_predictions[i][len(word_ids) - 1] = -1e4

                        start_signal = start_signal_predictions[i].argmax().item()
                        end_signal_predictions[i][: start_signal] = -1e4
                        end_signal_predictions[i][start_signal + 5: ] = -1e4
                        end_signal = end_signal_predictions[i].argmax().item()

                    if not args.beam_search:
                        space_splitted_tokens[word_ids[start_cause]] = '<ARG0>' + space_splitted_tokens[word_ids[start_cause]]
                        space_splitted_tokens[word_ids[end_cause]] = space_splitted_tokens[word_ids[end_cause]] + '</ARG0>'
                        space_splitted_tokens[word_ids[start_effect]] = '<ARG1>' + space_splitted_tokens[word_ids[start_effect]]
                        space_splitted_tokens[word_ids[end_effect]] = space_splitted_tokens[word_ids[end_effect]] + '</ARG1>'
                        
                        if has_signal:
                            space_splitted_tokens[word_ids[start_signal]] = '<SIG0>' + space_splitted_tokens[word_ids[start_signal]]
                            space_splitted_tokens[word_ids[end_signal]] = space_splitted_tokens[word_ids[end_signal]] + '</SIG0>'
                        
                        predictions.append([' '.join(space_splitted_tokens)] * 2)
                    else:
                        start_cause, end_cause, start_effect, end_effect = indices1

                        this_space_splitted_tokens = copy.deepcopy(space_splitted_tokens)
                        this_space_splitted_tokens[word_ids[start_cause]] = '<ARG0>' + this_space_splitted_tokens[word_ids[start_cause]]
                        this_space_splitted_tokens[word_ids[end_cause]] = this_space_splitted_tokens[word_ids[end_cause]] + '</ARG0>'
                        this_space_splitted_tokens[word_ids[start_effect]] = '<ARG1>' + this_space_splitted_tokens[word_ids[start_effect]]
                        this_space_splitted_tokens[word_ids[end_effect]] = this_space_splitted_tokens[word_ids[end_effect]] + '</ARG1>'

                        if has_signal:
                            this_space_splitted_tokens[word_ids[start_signal]] = '<SIG0>' + this_space_splitted_tokens[word_ids[start_signal]]
                            this_space_splitted_tokens[word_ids[end_signal]] = this_space_splitted_tokens[word_ids[end_signal]] + '</SIG0>'
                        generated_sentence_1 = ' '.join(this_space_splitted_tokens)

                        start_cause, end_cause, start_effect, end_effect = indices2

                        this_space_splitted_tokens = copy.deepcopy(space_splitted_tokens)
                        this_space_splitted_tokens[word_ids[start_cause]] = '<ARG0>' + this_space_splitted_tokens[word_ids[start_cause]]
                        this_space_splitted_tokens[word_ids[end_cause]] = this_space_splitted_tokens[word_ids[end_cause]] + '</ARG0>'
                        this_space_splitted_tokens[word_ids[start_effect]] = '<ARG1>' + this_space_splitted_tokens[word_ids[start_effect]]
                        this_space_splitted_tokens[word_ids[end_effect]] = this_space_splitted_tokens[word_ids[end_effect]] + '</ARG1>'
                        
                        if has_signal:
                            this_space_splitted_tokens[word_ids[start_signal]] = '<SIG0>' + this_space_splitted_tokens[word_ids[start_signal]]
                            this_space_splitted_tokens[word_ids[end_signal]] = this_space_splitted_tokens[word_ids[end_signal]] + '</SIG0>'
                        generated_sentence_2 = ' '.join(this_space_splitted_tokens)

                        predictions.append([generated_sentence_1, generated_sentence_2])
