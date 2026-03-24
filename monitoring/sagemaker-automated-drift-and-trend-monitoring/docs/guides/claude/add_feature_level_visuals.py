"""
Feature-Level Drift Analysis - Code to Add to Notebook

Add this as new cells in your 3_governance_dashboard.ipynb notebook:
1. After Cell 17 (feature drift dataset): Add feature-level dataset
2. After your last visual cell: Add feature-level visuals
3. Update analysis/dashboard definitions to include Sheet 4

USAGE:
1. Copy sections below into new notebook cells
2. Run in order
3. Analysis and Dashboard will be updated with new sheet
"""

# =============================================================================
# CELL 1: Create Feature-Level Dataset (insert after Cell 17)
# =============================================================================

# Feature-level drift dataset (from Athena view 'feature_drift_detail')
# Shows individual feature drift scores across monitoring runs

FEATURE_LEVEL_DATASET_ID = 'fraud-governance-feature-level-dataset'
FEATURE_LEVEL_DATASET_NAME = 'Fraud Governance - Feature Level Drift'

feature_level_physical_table = {
    'feature-level-view': {
        'RelationalTable': {
            'DataSourceArn': DATASOURCE_ARN,
            'Catalog': 'AwsDataCatalog',
            'Schema': ATHENA_DATABASE,
            'Name': 'feature_drift_detail',  # Athena view created earlier
            'InputColumns': [
                {'Name': 'monitoring_run_id', 'Type': 'STRING'},
                {'Name': 'monitoring_timestamp', 'Type': 'DATETIME'},
                {'Name': 'model_version', 'Type': 'STRING'},
                {'Name': 'endpoint_name', 'Type': 'STRING'},
                {'Name': 'data_drift_detected', 'Type': 'BIT'},
                {'Name': 'drifted_columns_count', 'Type': 'INTEGER'},
                {'Name': 'drifted_columns_share', 'Type': 'DECIMAL'},
                {'Name': 'baseline_roc_auc', 'Type': 'DECIMAL'},
                {'Name': 'current_roc_auc', 'Type': 'DECIMAL'},
                {'Name': 'feature_name', 'Type': 'STRING'},
                {'Name': 'drift_score', 'Type': 'DECIMAL'},
                {'Name': 'drift_severity', 'Type': 'STRING'},
                {'Name': 'drift_detected', 'Type': 'BIT'},
            ],
        }
    }
}

feature_level_logical_table = {
    'feature-level-logical': {
        'Alias': 'Feature Level Drift',
        'Source': {'PhysicalTableId': 'feature-level-view'},
    }
}

feature_level_dset_common = dict(
    AwsAccountId=ACCOUNT_ID,
    DataSetId=FEATURE_LEVEL_DATASET_ID,
    Name=FEATURE_LEVEL_DATASET_NAME,
    PhysicalTableMap=feature_level_physical_table,
    LogicalTableMap=feature_level_logical_table,
    ImportMode='DIRECT_QUERY',
)

try:
    quicksight.describe_data_set(AwsAccountId=ACCOUNT_ID, DataSetId=FEATURE_LEVEL_DATASET_ID)
    print('Updating existing feature-level dataset...')
    resp = quicksight.update_data_set(**feature_level_dset_common)
except ClientError as e:
    if e.response['Error']['Code'] == 'ResourceNotFoundException':
        print('Creating new feature-level dataset...')
        resp = quicksight.create_data_set(
            **feature_level_dset_common,
            Permissions=[{'Principal': QS_PRINCIPAL, 'Actions': DSET_ACTIONS}],
        )
    else:
        raise

FEATURE_LEVEL_DATASET_ARN = resp['Arn']
print(f'✓ Feature-level dataset: {FEATURE_LEVEL_DATASET_ARN}')

# =============================================================================
# CELL 2: Define Feature-Level Visuals (insert after last visual cell)
# =============================================================================

# Helper: column identifier for feature-level dataset
def flcol(name):
    return {'DataSetIdentifier': 'feature-level-ds', 'ColumnName': name}

# V17: Feature Drift Timeline (line chart - compare selected features over time)
v17_feature_timeline = {
    'LineChartVisual': {
        'VisualId': 'v17-feature-timeline',
        'Title': {'Visibility': 'VISIBLE', 'FormatText': {'PlainText': 'Feature Drift Over Time'}},
        'ChartConfiguration': {
            'FieldWells': {
                'LineChartAggregatedFieldWells': {
                    'Category': [{'DateDimensionField': {'FieldId': 'time', 'Column': flcol('monitoring_timestamp'), 'DateGranularity': 'DAY'}}],
                    'Values': [{'NumericalMeasureField': {'FieldId': 'drift', 'Column': flcol('drift_score'), 'AggregationFunction': {'SimpleNumericalAggregation': 'AVERAGE'}}}],
                    'Colors': [{'CategoricalDimensionField': {'FieldId': 'feature', 'Column': flcol('feature_name')}}],
                }
            },
            'SortConfiguration': {},
            'Type': 'LINE',
            'Legend': {'Visibility': 'VISIBLE', 'Position': 'RIGHT'},
            'PrimaryYAxisDisplayOptions': {'AxisOptions': {'AxisLineVisibility': 'VISIBLE'}},
        }
    }
}

# V18: Top Drifting Features (horizontal bar - avg drift by feature)
v18_top_features = {
    'BarChartVisual': {
        'VisualId': 'v18-top-features',
        'Title': {'Visibility': 'VISIBLE', 'FormatText': {'PlainText': 'Top 15 Drifting Features'}},
        'ChartConfiguration': {
            'FieldWells': {
                'BarChartAggregatedFieldWells': {
                    'Category': [{'CategoricalDimensionField': {'FieldId': 'feature', 'Column': flcol('feature_name')}}],
                    'Values': [{'NumericalMeasureField': {'FieldId': 'avg-drift', 'Column': flcol('drift_score'), 'AggregationFunction': {'SimpleNumericalAggregation': 'AVERAGE'}}}],
                }
            },
            'SortConfiguration': {
                'CategoryItemsLimit': {'OtherCategories': 'INCLUDE', 'ItemsLimit': 15},
                'CategorySort': [{'FieldSort': {'FieldId': 'avg-drift', 'Direction': 'DESC'}}],
            },
            'Orientation': 'HORIZONTAL',
            'BarsArrangement': 'CLUSTERED',
            'Legend': {'Visibility': 'HIDDEN'},
        }
    }
}

# V19: Drift Severity Distribution (stacked bar by severity)
v19_severity_dist = {
    'BarChartVisual': {
        'VisualId': 'v19-severity-dist',
        'Title': {'Visibility': 'VISIBLE', 'FormatText': {'PlainText': 'Drift Severity by Feature (Top 15)'}},
        'ChartConfiguration': {
            'FieldWells': {
                'BarChartAggregatedFieldWells': {
                    'Category': [{'CategoricalDimensionField': {'FieldId': 'feature', 'Column': flcol('feature_name')}}],
                    'Values': [{'NumericalMeasureField': {'FieldId': 'count', 'Column': flcol('drift_score'), 'AggregationFunction': {'SimpleNumericalAggregation': 'COUNT'}}}],
                    'Colors': [{'CategoricalDimensionField': {'FieldId': 'severity', 'Column': flcol('drift_severity')}}],
                }
            },
            'SortConfiguration': {
                'CategoryItemsLimit': {'OtherCategories': 'INCLUDE', 'ItemsLimit': 15},
                'CategorySort': [{'FieldSort': {'FieldId': 'count', 'Direction': 'DESC'}}],
            },
            'Orientation': 'HORIZONTAL',
            'BarsArrangement': 'STACKED',
            'Legend': {'Visibility': 'VISIBLE', 'Position': 'RIGHT'},
        }
    }
}

# V20: Feature Drift Detail Table
v20_feature_detail = {
    'TableVisual': {
        'VisualId': 'v20-feature-detail',
        'Title': {'Visibility': 'VISIBLE', 'FormatText': {'PlainText': 'Feature Drift Details'}},
        'ChartConfiguration': {
            'FieldWells': {
                'TableAggregatedFieldWells': {
                    'GroupBy': [
                        {'DateDimensionField': {'FieldId': 'time', 'Column': flcol('monitoring_timestamp'), 'DateGranularity': 'MINUTE'}},
                        {'CategoricalDimensionField': {'FieldId': 'feature', 'Column': flcol('feature_name')}},
                        {'CategoricalDimensionField': {'FieldId': 'severity', 'Column': flcol('drift_severity')}},
                        {'CategoricalDimensionField': {'FieldId': 'model', 'Column': flcol('model_version')}},
                    ],
                    'Values': [
                        {'NumericalMeasureField': {'FieldId': 'drift', 'Column': flcol('drift_score'), 'AggregationFunction': {'SimpleNumericalAggregation': 'AVERAGE'}}},
                    ],
                }
            },
            'SortConfiguration': {
                'RowSort': [{'FieldSort': {'FieldId': 'time', 'Direction': 'DESC'}}],
            },
        }
    }
}

# V21: Most Problematic Feature KPI
v21_worst_feature = {
    'KPIVisual': {
        'VisualId': 'v21-worst-feature',
        'Title': {'Visibility': 'VISIBLE', 'FormatText': {'PlainText': 'Highest Drifting Feature'}},
        'ChartConfiguration': {
            'FieldWells': {
                'Values': [{'NumericalMeasureField': {'FieldId': 'max-drift', 'Column': flcol('drift_score'), 'AggregationFunction': {'SimpleNumericalAggregation': 'MAX'}}}],
            },
            'SortConfiguration': {},
        }
    }
}

# V22: Drift Heatmap (pivot table)
v22_drift_heatmap = {
    'PivotTableVisual': {
        'VisualId': 'v22-drift-heatmap',
        'Title': {'Visibility': 'VISIBLE', 'FormatText': {'PlainText': 'Feature Drift Heatmap'}},
        'ChartConfiguration': {
            'FieldWells': {
                'PivotTableAggregatedFieldWells': {
                    'Rows': [{'CategoricalDimensionField': {'FieldId': 'feature', 'Column': flcol('feature_name')}}],
                    'Columns': [{'DateDimensionField': {'FieldId': 'date', 'Column': flcol('monitoring_timestamp'), 'DateGranularity': 'DAY'}}],
                    'Values': [{'NumericalMeasureField': {'FieldId': 'drift', 'Column': flcol('drift_score'), 'AggregationFunction': {'SimpleNumericalAggregation': 'AVERAGE'}}}],
                }
            },
            'SortConfiguration': {},
        }
    }
}

FEATURE_LEVEL_VISUALS = [
    v17_feature_timeline,
    v18_top_features,
    v19_severity_dist,
    v20_feature_detail,
    v21_worst_feature,
    v22_drift_heatmap
]

print(f'Defined {len(FEATURE_LEVEL_VISUALS)} feature-level visuals')

# =============================================================================
# CELL 3: Update Analysis Definition (modify existing analysis cell)
# =============================================================================

# Add this to your existing analysis_definition in the analysis creation cell:

# In DataSetIdentifierDeclarations, add:
# {'Identifier': 'feature-level-ds', 'DataSetArn': FEATURE_LEVEL_DATASET_ARN},

# In Sheets, add:
# {
#     'SheetId': 'governance-sheet-4',
#     'Name': 'Feature Drift Detail',
#     'Visuals': FEATURE_LEVEL_VISUALS,
# },

# Full updated analysis_definition:
analysis_definition = {
    'DataSetIdentifierDeclarations': [
        {'Identifier': 'inference-ds', 'DataSetArn': DATASET_ARN},
        {'Identifier': 'drift-ds', 'DataSetArn': DRIFT_DATASET_ARN},
        {'Identifier': 'feature-drift-ds', 'DataSetArn': FEATURE_DRIFT_DATASET_ARN},
        {'Identifier': 'feature-level-ds', 'DataSetArn': FEATURE_LEVEL_DATASET_ARN},  # NEW
    ],
    'Sheets': [
        {
            'SheetId': 'governance-sheet-1',
            'Name': 'Inference Monitoring',
            'Visuals': INFERENCE_VISUALS,
        },
        {
            'SheetId': 'governance-sheet-2',
            'Name': 'Drift Trend Analysis',
            'Visuals': DRIFT_VISUALS,
        },
        {
            'SheetId': 'governance-sheet-3',
            'Name': 'Feature Drift Analysis',
            'Visuals': FEATURE_DRIFT_VISUALS,
        },
        {
            'SheetId': 'governance-sheet-4',
            'Name': 'Feature Drift Detail',  # NEW SHEET
            'Visuals': FEATURE_LEVEL_VISUALS,
        },
    ],
}

# Same for dashboard_definition in the dashboard creation cell

print("✓ Analysis and Dashboard definitions updated with Sheet 4")
