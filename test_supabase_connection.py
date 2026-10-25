
from app.supabase_client import get_supabase_client

client = get_supabase_client()

result = client.table("documents").select("id").limit(1).execute()

print("Supabase connection successful.")
print("Documents table is accessible.")
