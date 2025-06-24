export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[];

export type Database = {
  public: {
    Tables: {
      agent_logs: {
        Row: {
          agent_input: Json | null;
          agent_output: Json | null;
          agent_type: string;
          conversation_id: string | null;
          created_at: string | null;
          error_message: string | null;
          id: string;
          message_id: string | null;
          processing_time_ms: number | null;
          status: string | null;
        };
        Insert: {
          agent_input?: Json | null;
          agent_output?: Json | null;
          agent_type: string;
          conversation_id?: string | null;
          created_at?: string | null;
          error_message?: string | null;
          id?: string;
          message_id?: string | null;
          processing_time_ms?: number | null;
          status?: string | null;
        };
        Update: {
          agent_input?: Json | null;
          agent_output?: Json | null;
          agent_type?: string;
          conversation_id?: string | null;
          created_at?: string | null;
          error_message?: string | null;
          id?: string;
          message_id?: string | null;
          processing_time_ms?: number | null;
          status?: string | null;
        };
        Relationships: [
          {
            foreignKeyName: "agent_logs_conversation_id_fkey";
            columns: ["conversation_id"];
            isOneToOne: false;
            referencedRelation: "conversations";
            referencedColumns: ["id"];
          },
          {
            foreignKeyName: "agent_logs_message_id_fkey";
            columns: ["message_id"];
            isOneToOne: false;
            referencedRelation: "messages";
            referencedColumns: ["id"];
          }
        ];
      };
      conversations: {
        Row: {
          created_at: string | null;
          id: string;
          metadata: Json | null;
          title: string | null;
          updated_at: string | null;
          user_id: string | null;
        };
        Insert: {
          created_at?: string | null;
          id?: string;
          metadata?: Json | null;
          title?: string | null;
          updated_at?: string | null;
          user_id?: string | null;
        };
        Update: {
          created_at?: string | null;
          id?: string;
          metadata?: Json | null;
          title?: string | null;
          updated_at?: string | null;
          user_id?: string | null;
        };
        Relationships: [];
      };
      documents: {
        Row: {
          content: string;
          created_at: string | null;
          file_size: number | null;
          file_type: string | null;
          id: string;
          metadata: Json | null;
          title: string;
          updated_at: string | null;
        };
        Insert: {
          content: string;
          created_at?: string | null;
          file_size?: number | null;
          file_type?: string | null;
          id?: string;
          metadata?: Json | null;
          title: string;
          updated_at?: string | null;
        };
        Update: {
          content?: string;
          created_at?: string | null;
          file_size?: number | null;
          file_type?: string | null;
          id?: string;
          metadata?: Json | null;
          title?: string;
          updated_at?: string | null;
        };
        Relationships: [];
      };
      embeddings: {
        Row: {
          chunk_index: number;
          chunk_metadata: Json | null;
          chunk_text: string;
          created_at: string | null;
          document_id: string | null;
          embedding: string | null;
          id: string;
        };
        Insert: {
          chunk_index: number;
          chunk_metadata?: Json | null;
          chunk_text: string;
          created_at?: string | null;
          document_id?: string | null;
          embedding?: string | null;
          id?: string;
        };
        Update: {
          chunk_index?: number;
          chunk_metadata?: Json | null;
          chunk_text?: string;
          created_at?: string | null;
          document_id?: string | null;
          embedding?: string | null;
          id?: string;
        };
        Relationships: [
          {
            foreignKeyName: "embeddings_document_id_fkey";
            columns: ["document_id"];
            isOneToOne: false;
            referencedRelation: "documents";
            referencedColumns: ["id"];
          }
        ];
      };
      messages: {
        Row: {
          agent_data: Json | null;
          content: string;
          conversation_id: string | null;
          created_at: string | null;
          id: string;
          metadata: Json | null;
          role: string;
        };
        Insert: {
          agent_data?: Json | null;
          content: string;
          conversation_id?: string | null;
          created_at?: string | null;
          id?: string;
          metadata?: Json | null;
          role: string;
        };
        Update: {
          agent_data?: Json | null;
          content?: string;
          conversation_id?: string | null;
          created_at?: string | null;
          id?: string;
          metadata?: Json | null;
          role?: string;
        };
        Relationships: [
          {
            foreignKeyName: "messages_conversation_id_fkey";
            columns: ["conversation_id"];
            isOneToOne: false;
            referencedRelation: "conversations";
            referencedColumns: ["id"];
          }
        ];
      };
    };
    Views: {
      [_ in never]: never;
    };
    Functions: {
      [_ in never]: never;
    };
    Enums: {
      [_ in never]: never;
    };
    CompositeTypes: {
      [_ in never]: never;
    };
  };
};

type DefaultSchema = Database[Extract<keyof Database, "public">];

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof Database },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof Database;
  }
    ? keyof (Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        Database[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never
> = DefaultSchemaTableNameOrOptions extends { schema: keyof Database }
  ? (Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      Database[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R;
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
      DefaultSchema["Views"])
  ? (DefaultSchema["Tables"] &
      DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
      Row: infer R;
    }
    ? R
    : never
  : never;

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof Database },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof Database;
  }
    ? keyof Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never
> = DefaultSchemaTableNameOrOptions extends { schema: keyof Database }
  ? Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I;
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
  ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
      Insert: infer I;
    }
    ? I
    : never
  : never;

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof Database },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof Database;
  }
    ? keyof Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never
> = DefaultSchemaTableNameOrOptions extends { schema: keyof Database }
  ? Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U;
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
  ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
      Update: infer U;
    }
    ? U
    : never
  : never;
