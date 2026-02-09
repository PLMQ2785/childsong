음악생성
curl --request POST \
  --url https://api.sunoapi.org/api/v1/generate \
  --header 'Authorization: Bearer <token>' \
  --header 'Content-Type: application/json' \
  --data '
{
  "customMode": true,
  "instrumental": true,
  "model": "V4_5ALL",
  "callBackUrl": "https://api.example.com/callback",
  "prompt": "A calm and relaxing piano track with soft melodies",
  "style": "Classical",
  "title": "Peaceful Piano Meditation",
  "personaId": "persona_123",
  "negativeTags": "Heavy Metal, Upbeat Drums",
  "vocalGender": "m",
  "styleWeight": 0.65,
  "weirdnessConstraint": 0.65,
  "audioWeight": 0.65
}
'

음악정보 가져오기
curl --request GET \
  --url https://api.sunoapi.org/api/v1/generate/record-info \
  --header 'Authorization: Bearer <token>'

음악 늘리기
curl --request POST \
  --url https://api.sunoapi.org/api/v1/generate/extend \
  --header 'Authorization: Bearer <token>' \
  --header 'Content-Type: application/json' \
  --data '
{
  "defaultParamFlag": true,
  "audioId": "e231****-****-****-****-****8cadc7dc",
  "model": "V4_5ALL",
  "callBackUrl": "https://api.example.com/callback",
  "prompt": "Extend the music with more relaxing notes",
  "style": "Classical",
  "title": "Peaceful Piano Extended",
  "continueAt": 60,
  "personaId": "persona_123",
  "negativeTags": "Relaxing Piano",
  "vocalGender": "m",
  "styleWeight": 0.65,
  "weirdnessConstraint": 0.65,
  "audioWeight": 0.65
}
'

