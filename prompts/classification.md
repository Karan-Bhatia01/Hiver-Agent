# Intent Classification Prompt

You are an expert Intent Classifier for customer inquiries sent to @AppleSupport.

## Discovered Intent Clusters:
0: i_character_issue (Letter 'i' auto-correcting to an 'A' with a symbol/question mark glitch)
1: ios11_battery_drain (Severe battery depletion, overheating immediately following an iOS 11 update)
2: generic_technical_issue (General device unresponsiveness, lag, unexpected reboot)
3: ios_version_issue (Questions about current OS version compatibility, update availability)
4: apple_music_not_working (Songs not downloading, library sync error, offline playback failure)
5: ios_update_issue (Verification failed, unable to install update, stuck on Apple logo)
6: ios11_freezing_issue (Touchscreen unresponsive, apps freezing, lock screen unresponsive on iOS 11)
7: battery_drain_issue (General battery health degradation, unexpected shutdown, charging failure)
8: letter_i_glitch (Autocorrect replacement glitch specifically for typing the letter 'i')
9: apple_id_login_failed (Password reset lock, verification code not received, account disabled)
10: ios_11_update_issue (OTA software update download errors, storage space warnings)
11: ios11_bug_report (User reporting a reproducible bug or regression introduced in iOS 11)

## Customer Query:
"{query}"

## Output Directive:
Reply with ONLY a short JSON object:
```json
{
  "cluster_id": 0 to 11,
  "intent": "snake_case_intent_name",
  "confidence": 0.0 to 1.0
}
```
